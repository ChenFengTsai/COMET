#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import ruamel.yaml as yaml

import tools
import models_pretrain


# =============================================================================
# Run config loading (from same run dir as student_dir)
# =============================================================================

def _load_yaml_or_json(path: pathlib.Path) -> dict:
    suf = path.suffix.lower()
    if suf in [".yaml", ".yml"]:
        return yaml.safe_load(path.read_text())
    if suf == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path}")


def load_run_config_from_student_dir(student_dir: str) -> dict | None:
    """
    student_dir usually like: /.../run/train_eps or /.../run/episodes
    Looks for common run config names in run_dir = parent(student_dir).
    """
    run_dir = pathlib.Path(student_dir).resolve().parent
    candidates = [
        "config.yaml", "config.yml", "config.json",
        "args.yaml", "args.yml", "args.json",
        "params.yaml", "params.yml", "params.json",
    ]
    for name in candidates:
        p = run_dir / name
        if p.exists():
            print(f"[INFO] Loading run config from {p}")
            return _load_yaml_or_json(p)
    print("[INFO] No run config found next to student_dir.")
    return None


# =============================================================================
# Episode loading (CORRECT for your repo: tools.load_episodes .npz)
# =============================================================================

def load_npz_episodes_as_list(directory: str,
                              limit: Optional[int] = None,
                              reverse: bool = True) -> List[Dict[str, Any]]:
    """
    tools.load_episodes returns dict[path -> episode_dict], episode_dict values are numpy arrays.
    Convert to list in deterministic order.
    """
    eps = tools.load_episodes(directory, limit=limit, reverse=reverse)
    keys = sorted(eps.keys())
    return [eps[k] for k in keys]


def standardize_episode_arrays(ep: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      image: (T,H,W,C)
      action:(T,A)

    Handles:
      - keys: image/observation, action/actions
      - (T,C,H,W) -> (T,H,W,C)
      - (B,T,...) -> take B=0
    """
    if "image" in ep:
        img = ep["image"]
    elif "observation" in ep:
        img = ep["observation"]
    else:
        raise KeyError("Episode missing 'image' (or 'observation').")

    if "action" in ep:
        act = ep["action"]
    elif "actions" in ep:
        act = ep["actions"]
    else:
        raise KeyError("Episode missing 'action' (or 'actions').")

    img = np.asarray(img)
    act = np.asarray(act)

    # (B,T,...) -> (T,...)
    if img.ndim == 5:
        img = img[0]
    if act.ndim == 3:
        act = act[0]

    # (T,C,H,W) -> (T,H,W,C)
    if img.ndim == 4 and img.shape[1] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (0, 2, 3, 1))

    if img.ndim != 4 or act.ndim != 2:
        raise ValueError(f"Bad episode shapes: image={img.shape}, action={act.shape}")

    return img, act


# =============================================================================
# Config fixups (mirror training-time conversions)
# =============================================================================

def make_posthoc_schedules_callable(cfg) -> None:
    """
    Training wraps schedule strings into callables. Post-hoc needs the same.
    Step is irrelevant here; use step=0.
    """
    step0 = 0

    def ensure_callable(name: str, fallback_from: Optional[str] = None):
        if hasattr(cfg, name) and callable(getattr(cfg, name)):
            return

        if hasattr(cfg, name) and isinstance(getattr(cfg, name), str):
            s = getattr(cfg, name)
            setattr(cfg, f"{name}_str", s)
            setattr(cfg, name, (lambda x=s: tools.schedule(x, step0)))
            return

        if fallback_from and hasattr(cfg, fallback_from) and isinstance(getattr(cfg, fallback_from), str):
            s = getattr(cfg, fallback_from)
            setattr(cfg, name, (lambda x=s: tools.schedule(x, step0)))

    ensure_callable("moe_temperature", fallback_from="moe_temperature_str")
    ensure_callable("actor_entropy", fallback_from="actor_entropy_str")
    ensure_callable("actor_state_entropy", fallback_from="actor_state_entropy_str")
    ensure_callable("imag_gradient_mix", fallback_from="imag_gradient_mix_str")


def make_act_callable_only(cfg) -> None:
    """
    Fix: networks.ConvEncoder expects cfg.act callable (act()).
    Keep cfg.teacher_act as string because models_distill.py parses it via .split().
    """
    import torch.nn as nn

    def parse_act(val):
        if callable(val):
            return val
        if isinstance(val, str):
            if "ELU" in val:
                return nn.ELU
            if "LeakyReLU" in val:
                return nn.LeakyReLU
            if "SiLU" in val or "Swish" in val:
                return nn.SiLU
            if "ReLU" in val:
                return nn.ReLU
            raise ValueError(f"Unknown act string: {val}")
        return val

    if hasattr(cfg, "act"):
        cfg.act = parse_act(cfg.act)


def normalize_tuple_fields(cfg) -> None:
    """
    Fix: code does (channels,) + cfg.size; JSON makes cfg.size a list.
    """
    if hasattr(cfg, "size") and isinstance(cfg.size, list):
        cfg.size = tuple(cfg.size)
    if hasattr(cfg, "teacher_size") and isinstance(cfg.teacher_size, list):
        cfg.teacher_size = tuple(cfg.teacher_size)


def prepare_config_for_posthoc(cfg) -> None:
    make_posthoc_schedules_callable(cfg)
    make_act_callable_only(cfg)
    normalize_tuple_fields(cfg)


# =============================================================================
# Teacher loading (CORRECT: WorldModelStudent.load_teacher)
# =============================================================================

@dataclass
class DummyActionSpace:
    low: np.ndarray
    high: np.ndarray


def infer_action_space_from_episode(ep: Dict[str, Any]) -> DummyActionSpace:
    _, act = standardize_episode_arrays(ep)
    dim = int(act.shape[-1])
    return DummyActionSpace(
        low=-np.ones((dim,), dtype=np.float32),
        high=np.ones((dim,), dtype=np.float32),
    )


def load_teacher_modules(config, example_episode: Dict[str, Any]):
    """Load teacher encoder+dynamics from a *pretraining* teacher checkpoint.

    The OOD script originally initialized a distillation student and called
    `wm.load_teacher(...)`. That breaks when the provided checkpoint was saved
    from the pretraining teacher (see models_pretrain.py). Here we directly
    instantiate `models_pretrain.WorldModelTeacher` and load its weights.
    """
    if not getattr(config, "teacher_model_path", None):
        raise ValueError(
            "teacher_model_path is empty. Pass --teacher_model_path /path/to/teacher.pt"
        )

    action_space = infer_action_space_from_episode(example_episode)

    def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            # If it already looks like a plain state_dict.
            if len(obj) and all(torch.is_tensor(v) for v in obj.values()):
                return obj  # type: ignore[return-value]
            for k in (
                "state_dict",
                "model_state_dict",
                "model",
                "teacher",
                "world_model",
                "wm",
                "params",
            ):
                sd = obj.get(k, None)
                if isinstance(sd, dict) and len(sd) and all(torch.is_tensor(v) for v in sd.values()):
                    return sd  # type: ignore[return-value]
            # Sometimes nested one level deeper.
            for v in obj.values():
                if isinstance(v, dict) and len(v) and all(torch.is_tensor(t) for t in v.values()):
                    return v  # type: ignore[return-value]
        raise ValueError(
            "Unrecognized teacher checkpoint format. Expected a state_dict or a dict containing one."
        )

    teacher = models_pretrain.WorldModelTeacher(
        step=0,
        config=config,
        action_space=action_space,
    ).to(config.device)

    ckpt = torch.load(config.teacher_model_path, map_location=config.device)
    teacher_sd = _extract_state_dict(ckpt)

    missing, unexpected = teacher.load_state_dict(teacher_sd, strict=False)
    if missing:
        print(f"[WARN] Teacher checkpoint missing {len(missing)} keys (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Teacher checkpoint has {len(unexpected)} unexpected keys (showing up to 10): {unexpected[:10]}")

    teacher.eval()
    enc = teacher.encoder.eval()
    dyn = teacher.dynamics.eval()
    return enc, dyn


# =============================================================================
# Embedding extraction (teacher conditioned on label)
# =============================================================================

def fit_gaussian(x: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, (x.shape[0] - 1))
    cov = cov + eps * torch.eye(cov.shape[0], device=x.device, dtype=x.dtype)
    inv = torch.linalg.inv(cov)
    return mu, inv


def mahalanobis(x: torch.Tensor, mu: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    d = x - mu
    return torch.sum((d @ inv) * d, dim=1)


@torch.no_grad()
def episode_to_embedding(
    enc: torch.nn.Module,
    dyn: torch.nn.Module,
    ep: Dict[str, Any],
    device: torch.device,
    images_already_normalized: bool,
    label: int,                    # IMPORTANT: label must be int (no None)
    skip_first: bool = True,
) -> torch.Tensor:
    """
    Returns (D,) embedding for an episode, extracted from teacher WM conditioned on label.
    This avoids the conditional-dim mismatch you hit.
    """
    img, act = standardize_episode_arrays(ep)

    obs = torch.from_numpy(img).to(device).unsqueeze(0).float()  # (1,T,H,W,C)
    act = torch.from_numpy(act).to(device).unsqueeze(0).float()  # (1,T,A)

    if not images_already_normalized:
        obs = obs / 255.0 - 0.5

    data = {"image": obs}

    # Encoder may accept label (MoE)
    try:
        embed = enc(data, label=label)
    except TypeError:
        embed = enc(data)

    # Dynamics may accept label
    try:
        post, _ = dyn.observe(embed, act, label=label)
    except TypeError:
        post, _ = dyn.observe(embed, act)

    feat = dyn.get_feat(post)  # (1,T,D)
    if skip_first and feat.shape[1] > 1:
        feat = feat[:, 1:]
    emb = feat.mean(dim=1).squeeze(0)  # (D,)
    return emb


# =============================================================================
# Metrics A/B: best-of-labels and mixture
# =============================================================================

@torch.no_grad()
def score_best_of_labels(
    enc, dyn,
    episode: dict,
    stats_per_label: List[Dict[str, torch.Tensor]],
    device: torch.device,
    images_already_normalized: bool,
) -> Tuple[float, int]:
    """
    Metric A:
      score_min = min_i d_i(f_i(x))
    Returns:
      score_min, argmin_label
    """
    best = float("inf")
    best_label = -1
    for label, st in enumerate(stats_per_label):
        emb = episode_to_embedding(enc, dyn, episode, device, images_already_normalized, label).unsqueeze(0)
        d2 = mahalanobis(emb, st["mu"], st["inv"]).mean().item()
        if d2 < best:
            best = d2
            best_label = label
    return best, best_label


@torch.no_grad()
def score_mixture(
    enc, dyn,
    episode: dict,
    stats_per_label: List[Dict[str, torch.Tensor]],
    device: torch.device,
    images_already_normalized: bool,
    pi: Optional[torch.Tensor] = None,
) -> float:
    """
    Metric B (mixture):
      score = -log sum_i pi_i * exp(-d_i(f_i(x)))
    Lower = more in-distribution.
    """
    K = len(stats_per_label)
    if pi is None:
        pi = torch.full((K,), 1.0 / K, device=device, dtype=torch.float32)
    else:
        pi = pi.to(device).float()
        pi = pi / pi.sum()

    d_list = []
    for label, st in enumerate(stats_per_label):
        emb = episode_to_embedding(enc, dyn, episode, device, images_already_normalized, label).unsqueeze(0)
        d2 = mahalanobis(emb, st["mu"], st["inv"]).mean()  # tensor
        d_list.append(d2)

    d = torch.stack(d_list, dim=0)  # (K,)
    mixture_log = torch.logsumexp(torch.log(pi) - d, dim=0)
    return float((-mixture_log).item())


def build_threshold(scores: List[float], quantile: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("No scores to build threshold.")
    return float(np.quantile(arr, quantile))


def calibrate_thresholds(
    enc, dyn,
    heldout_eps: List[dict],
    stats_per_label: List[Dict[str, torch.Tensor]],
    device: torch.device,
    images_already_normalized: bool,
    quantile: float,
) -> Tuple[float, float]:
    """
    Builds thresholds for both metrics using held-out ID source episodes.
    Returns:
      thresh_min, thresh_mix
    """
    min_scores, mix_scores = [], []
    for ep in heldout_eps:
        smin, _ = score_best_of_labels(enc, dyn, ep, stats_per_label, device, images_already_normalized)
        smix = score_mixture(enc, dyn, ep, stats_per_label, device, images_already_normalized, pi=None)
        min_scores.append(smin)
        mix_scores.append(smix)

    return build_threshold(min_scores, quantile), build_threshold(mix_scores, quantile)


# =============================================================================
# Main pipeline
# =============================================================================

def run(config):
    prepare_config_for_posthoc(config)

    print(f"[INFO] Using device: {config.device}")
    device = torch.device(config.device)

    student_eps = load_npz_episodes_as_list(
        config.student_dir,
        limit=config.limit_student_episodes,
        reverse=True,
    )
    if not student_eps:
        raise RuntimeError(f"No student episodes found in: {config.student_dir}")
    print(f"[INFO] Loaded student episodes: {len(student_eps)} from {config.student_dir}")

    enc, dyn = load_teacher_modules(config, student_eps[0])
    print("[OK] Teacher loaded")

    # Source dir: get subdirectories (each subdirectory is a source task)
    source_root = pathlib.Path(config.source_dir)
    if not source_root.exists():
        raise ValueError(f"source_dir does not exist: {config.source_dir}")
    
    # Get all subdirectories in source_dir, sorted for deterministic order
    source_dirs = sorted([
        str(d) for d in source_root.iterdir() 
        if d.is_dir()
    ])
    
    K = len(source_dirs)
    if K <= 0:
        raise ValueError(f"No subdirectories found in source_dir: {config.source_dir}")

    print(f"[INFO] Found {K} source task dirs in {config.source_dir}")
    print(f"[INFO] Source task directories:")
    for i, d in enumerate(source_dirs):
        print(f"  [{i}] {d}")

    # Load source episodes per label
    source_eps_by_label: List[List[dict]] = []
    for label, d in enumerate(source_dirs):
        eps = load_npz_episodes_as_list(d, limit=config.limit_source_episodes, reverse=True)
        if not eps:
            raise RuntimeError(f"No episodes found in source dir: {d}")
        source_eps_by_label.append(eps)

    # Build per-label Gaussian stats using embeddings extracted with the SAME label
    print("\nSTEP 1: Fit per-label distributions from source embeddings")
    stats_per_label: List[Dict[str, torch.Tensor]] = []
    for label, eps_list in enumerate(source_eps_by_label):
        task_name = pathlib.Path(source_dirs[label]).name
        embs = []
        for ep in eps_list:
            emb = episode_to_embedding(
                enc, dyn, ep,
                device=device,
                images_already_normalized=bool(config.images_already_normalized),
                label=label,
                skip_first=True,
            )
            embs.append(emb.unsqueeze(0))
        X = torch.cat(embs, dim=0).to(device)  # (N,D)
        mu, inv = fit_gaussian(X)
        stats_per_label.append({"mu": mu, "inv": inv})
        print(f"  [{label}] {task_name:<30} embeds={tuple(X.shape)}")

    # Build heldout pool (ID) from each source label
    print("\nSTEP 2: Calibrate thresholds from held-out source episodes")
    g = torch.Generator()
    g.manual_seed(int(getattr(config, "seed", 0)))

    heldout_eps: List[dict] = []
    for label, eps_list in enumerate(source_eps_by_label):
        n = len(eps_list)
        m = int(n * float(config.heldout_frac))
        if m <= 0:
            continue
        idx = torch.randperm(n, generator=g).tolist()[:m]
        heldout_eps.extend([eps_list[i] for i in idx])

    if not heldout_eps:
        raise RuntimeError("No heldout episodes; increase data or heldout_frac.")

    thresh_min, thresh_mix = calibrate_thresholds(
        enc, dyn,
        heldout_eps=heldout_eps,
        stats_per_label=stats_per_label,
        device=device,
        images_already_normalized=bool(config.images_already_normalized),
        quantile=float(config.threshold_quantile),
    )
    print(f"[INFO] threshold_quantile={config.threshold_quantile}")
    print(f"[INFO] thresh_min (best-of-labels): {thresh_min:.6f}")
    print(f"[INFO] thresh_mix (mixture):        {thresh_mix:.6f}")

    # Classify student episodes with both metrics
    print("\nSTEP 3: Classify student episodes (both metrics)")
    print(f"\n{'Ep':<6} {'minScore':<12} {'minCls':<7} {'mixScore':<12} {'mixCls':<7} {'closest':<25}")
    print("-" * 80)

    results: Dict[int, Dict[str, Any]] = {}
    for i, ep in enumerate(student_eps):
        smin, argmin_label = score_best_of_labels(
            enc, dyn, ep, stats_per_label, device, bool(config.images_already_normalized)
        )
        smix = score_mixture(
            enc, dyn, ep, stats_per_label, device, bool(config.images_already_normalized), pi=None
        )

        is_ood_min = smin > thresh_min
        is_ood_mix = smix > thresh_mix

        closest_task = pathlib.Path(source_dirs[argmin_label]).name

        results[i] = {
            "min_score": float(smin),
            "min_is_ood": bool(is_ood_min),
            "mix_score": float(smix),
            "mix_is_ood": bool(is_ood_mix),
            "closest_label": int(argmin_label),
            "closest_task": closest_task,
        }

        print(f"{i:<6} {smin:<12.4f} {('OOD' if is_ood_min else 'ID'):<7} "
              f"{smix:<12.4f} {('OOD' if is_ood_mix else 'ID'):<7} {closest_task:<25}")

    # Save results
    out = pathlib.Path(config.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("OOD Results (two metrics)\n")
        f.write("=" * 80 + "\n")
        f.write(f"device: {config.device}\n")
        f.write(f"student_dir: {config.student_dir}\n")
        f.write(f"source_dir: {config.source_dir}\n")
        f.write(f"teacher_model_path: {config.teacher_model_path}\n")
        f.write(f"K (num source tasks): {K}\n")
        f.write(f"threshold_quantile: {config.threshold_quantile}\n")
        f.write(f"thresh_min: {thresh_min:.6f}\n")
        f.write(f"thresh_mix: {thresh_mix:.6f}\n\n")
        for i, r in results.items():
            f.write(f"Episode {i}:\n")
            for k, v in r.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\n[OK] Saved results -> {out}")


def build_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args, rest = parser.parse_known_args()

    cfg_path = pathlib.Path(__file__).resolve().parent / "configs.yaml"
    if not cfg_path.exists():
        alt = pathlib.Path.cwd() / "configs.yaml"
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError("Could not find configs.yaml next to script or in CWD.")

    configs = yaml.safe_load(cfg_path.read_text())

    defaults: Dict[str, Any] = {}
    for c in args.configs:
        defaults.update(configs[c])

    # early student_dir to locate run config
    early = argparse.ArgumentParser()
    early.add_argument("--student_dir", type=str, default=defaults.get("student_dir"))
    early_args, _ = early.parse_known_args(rest)

    if early_args.student_dir:
        run_cfg = load_run_config_from_student_dir(early_args.student_dir)
        if run_cfg:
            defaults.update(run_cfg)

    # Ensure important hyperparams always exist so CLI can override them
    defaults.setdefault("device", "cuda:0")
    defaults.setdefault("teacher_model_path", "")  # YOU WANT THIS AS HYPERPARAM
    defaults.setdefault("student_dir", early_args.student_dir or "student_data/episodes")
    
    # OOD defaults
    defaults.setdefault("source_dir", "")  # NEW: source directory path
    defaults.setdefault("output_file", "ood_results_coffee_push.txt")
    defaults.setdefault("heldout_frac", 0.2)
    defaults.setdefault("threshold_quantile", 0.95)
    defaults.setdefault("limit_source_episodes", None)
    defaults.setdefault("limit_student_episodes", None)
    defaults.setdefault("images_already_normalized", False)
    defaults.setdefault("seed", 0)

    parser = argparse.ArgumentParser()
    for k, v in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)

    cfg = parser.parse_args(rest)

    if not isinstance(cfg.device, str):
        cfg.device = str(cfg.device)
    
    # Validate source_dir
    if not cfg.source_dir:
        raise ValueError("--source_dir is required. Pass --source_dir /path/to/source")

    return cfg

def main():
    cfg = build_config()
    run(cfg)


if __name__ == "__main__":
    main()

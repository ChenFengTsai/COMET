import argparse
import pathlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import analysis_common as ac
import tools


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--teacher_model_path', required=True)
    p.add_argument('--teacher_encoder_mode', default='moe', choices=['moe', 'original_cnn'])
    p.add_argument('--task_a', default='pick_place')
    p.add_argument('--task_b', default='push')
    p.add_argument('--task_a_dir', required=True)
    p.add_argument('--task_b_dir', required=True)
    p.add_argument('--tag', default='model')
    p.add_argument('--outdir', default='./analysis_out')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_episodes', type=int, default=16,
                   help='trajectory windows per task (batch size)')
    p.add_argument('--batch_length', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def flatten_grads(grads):
    """Flatten a tuple of grad tensors (some may be None) into one 1D vector."""
    parts = []
    for g in grads:
        if g is not None:
            parts.append(g.reshape(-1))
    if not parts:
        return None
    return torch.cat(parts)


def model_loss_for_task(wm, config, data_np, label):
    """Recompute the teacher model loss for one task batch, WITH grad enabled.

    Mirrors WorldModelTeacher._train's loss construction (kl + head NLLs),
    but returns the scalar loss tensor so we can differentiate it w.r.t.
    chosen parameter groups. No optimizer step, no AMP autocast (we want
    clean float32 grads for cosine similarity).
    """
    data = wm.preprocess(data_np)
    embed = wm.encoder(data, label=label)
    post, prior = wm.dynamics.observe(embed, data['action'], label=label)

    kl_balance = tools.schedule(config.kl_balance, 0)
    kl_free = tools.schedule(config.kl_free, 0)
    kl_scale = tools.schedule(config.kl_scale, 0)
    kl_loss, _ = wm.dynamics.kl_loss(post, prior, config.kl_forward,
                                     kl_balance, kl_free, kl_scale)

    feat = wm.dynamics.get_feat(post)
    losses = {'kl': kl_loss}
    for name, head in wm.heads.items():
        grad_head = (name in config.grad_heads)
        feat_in = feat if grad_head else feat.detach()
        pred = head(feat_in, label=label)
        like = pred.log_prob(data[name])
        losses[name] = -torch.mean(like) * wm._scales.get(name, 1.0)
    return sum(losses.values())


def component_params(wm):
    """Macro-blocks to compare. Returns ordered dict-like list of (name, params)."""
    comps = [
        ('CNN Encoder', list(wm.encoder.parameters())),
        ('RSSM Core', list(wm.dynamics.parameters())),
    ]
    return comps


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overrides = {
        'teacher_encoder_mode': args.teacher_encoder_mode,
        'device': args.device,
        'teacher_model_path': args.teacher_model_path,
        # turn off AMP so grads are float32
        'precision': 32,
    }
    config = ac.load_config(args.configs, overrides)
    config.device = str(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    acts = ac.get_action_space(config)
    wm = ac.build_teacher(config, args.teacher_model_path, acts)

    # We need grads on the teacher params; eval() keeps norm/dropout fixed but
    # does not disable autograd.
    for p in wm.parameters():
        p.requires_grad_(True)

    label_a = ac.task_label(config, args.task_a)
    label_b = ac.task_label(config, args.task_b)
    print(f'[grad] {args.task_a} -> label {label_a}; {args.task_b} -> label {label_b}')

    data_a = ac.load_task_batch(config, args.task_a_dir,
                                args.n_episodes, args.batch_length, seed=args.seed)
    data_b = ac.load_task_batch(config, args.task_b_dir,
                                args.n_episodes, args.batch_length, seed=args.seed + 1)

    comps = component_params(wm)

    # Build the two scalar losses, each with its own graph.
    loss_a = model_loss_for_task(wm, config, data_a, label_a)
    loss_b = model_loss_for_task(wm, config, data_b, label_b)

    results = {}
    for cname, params in comps:
        params = [p for p in params if p.requires_grad]
        if len(params) == 0:
            print(f'[grad] {cname}: no params, skipping')
            continue
        # Separate grad tensors per task -> no .grad aliasing bug.
        g_a = torch.autograd.grad(loss_a, tuple(params),
                                  retain_graph=True, allow_unused=True)
        g_b = torch.autograd.grad(loss_b, tuple(params),
                                  retain_graph=True, allow_unused=True)
        fa = flatten_grads(g_a)
        fb = flatten_grads(g_b)
        if fa is None or fb is None:
            print(f'[grad] {cname}: all grads None (task may not route here), skipping')
            continue
        cos = F.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0)).item()
        results[cname] = cos
        print(f'[grad] {cname:14s} cosine similarity = {cos:+.4f}')

    # Save raw arrays
    npz_path = outdir / f'{args.tag}_grad_conflict.npz'
    np.savez(npz_path,
             components=np.array(list(results.keys())),
             cosine=np.array(list(results.values())),
             tag=args.tag, task_a=args.task_a, task_b=args.task_b)

    # Save human-readable to JSON
    json_path = outdir / f'{args.tag}_{args.task_a}_vs_{args.task_b}_grad_conflict.json'
    results_dict = {
        "tag": args.tag,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "cosine_similarities": results
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f'[grad] wrote arrays to {npz_path.name}')
    print(f'[grad] wrote readable values to {json_path.name}')

    # ---- Bar chart ----
    names = list(results.keys())
    vals = [results[n] for n in names]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(names, vals, color=colors, width=0.55)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('Gradient cosine similarity')
    ax.set_title(f'{args.tag} — gradient conflict\n'
                 f'{args.task_a} vs {args.task_b} at branching states')
    for i, v in enumerate(vals):
        ax.text(i, v + (0.04 if v >= 0 else -0.08), f'{v:+.2f}',
                ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f'{args.tag}_grad_conflict.png', dpi=160)
    plt.close(fig)
    print(f'[grad] done. Chart in {outdir}/{args.tag}_grad_conflict.png')


if __name__ == '__main__':
    main()
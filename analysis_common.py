"""
analysis_common.py

Shared utilities for the two COMET analysis scripts:
  - latent_projection.py        (PCA / t-SNE of the RSSM feature z_t)
  - gradient_conflict.py        (per-component gradient cosine similarity)

This module mirrors how dreamer_pretrain.py builds the model and how
models_pretrain.WorldModelTeacher consumes data, so the analyses use the
*exact* same forward path as training.

Place this file in your COMET repo root (next to models_pretrain.py,
networks.py, tools.py, configs.yaml) and run the scripts from there.
"""

import argparse
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml
import torch

sys.path.append(str(pathlib.Path(__file__).parent))

import models_pretrain
import tools
import wrappers


# --------------------------------------------------------------------------
# Config loading (replicates dreamer_pretrain.py's arg/config merge)
# --------------------------------------------------------------------------
def load_config(config_names, overrides=None):
    """Merge the named blocks from configs.yaml into a flat config namespace.

    config_names: list like ['defaults', 'metaworld', 'metaworld_teacher_moe_pretrain']
    overrides:    dict of {key: value} to force after merge (e.g. teacher_encoder_mode)
    """
    cfg_path = pathlib.Path(__file__).parent / 'configs.yaml'
    configs = yaml.safe_load(cfg_path.read_text())

    defaults = {}
    for name in config_names:
        defaults.update(configs[name])

    if overrides:
        defaults.update(overrides)

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    config = parser.parse_args([])

    # act is a string in yaml ('ELU'); WorldModelTeacher expects the class
    if isinstance(config.act, str):
        config.act = getattr(torch.nn, config.act)
    return config


# --------------------------------------------------------------------------
# Action space (the teacher only needs num_actions; build a dummy env once)
# --------------------------------------------------------------------------
def get_action_space(config):
    suite, task = config.task.split('_', 1)
    if suite == 'metaworld':
        task = '-'.join(task.split('_'))
        env = wrappers.MetaWorld(task, config.seed, config.action_repeat,
                                 config.size, config.camera)
        env = wrappers.NormalizeActions(env)
    elif suite == 'dmc':
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]
    return acts


# --------------------------------------------------------------------------
# Build teacher and load checkpoint
# --------------------------------------------------------------------------
def build_teacher(config, checkpoint_path, action_space):
    """Construct WorldModelTeacher and load a saved teacher_model.pt into it."""
    wm = models_pretrain.WorldModelTeacher(0, config, action_space).to(config.device)
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    # dreamer_pretrain.py saves agent._wm.state_dict() directly
    missing, unexpected = wm.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'[build_teacher] WARNING missing keys: {len(missing)} (showing 5) {missing[:5]}')
    if unexpected:
        print(f'[build_teacher] WARNING unexpected keys: {len(unexpected)} (showing 5) {unexpected[:5]}')
    wm.eval()
    return wm


# --------------------------------------------------------------------------
# Task name -> integer label, as used everywhere in the repo (label=task_id)
# --------------------------------------------------------------------------
def task_label(config, task_name):
    """Map a source-task name (e.g. 'pick_place') to its integer id.

    The label fed to encoder/dynamics is the *position* in source_tasks,
    which is exactly what dreamer_pretrain.py uses as task_id.
    """
    names = list(config.source_tasks)
    if task_name in names:
        return names.index(task_name)
    # tolerate hyphen/underscore variants
    norm = task_name.replace('-', '_')
    for i, n in enumerate(names):
        if n.replace('-', '_') == norm:
            return i
    raise ValueError(f"task '{task_name}' not in source_tasks={names}")


# --------------------------------------------------------------------------
# Load a batch of trajectory windows for one task, in the same dict format
# that WorldModelTeacher.preprocess expects (numpy, [B, T, ...]).
# --------------------------------------------------------------------------
def load_task_batch(config, task_dir, batch_size, batch_length, seed=0):
    episodes = tools.load_episodes(pathlib.Path(task_dir), limit=config.dataset_size)
    if len(episodes) == 0:
        raise RuntimeError(f'No .npz episodes found in {task_dir}')
    gen = tools.sample_episodes(episodes, batch_length, balance=False, seed=seed)
    batch = [next(gen) for _ in range(batch_size)]
    # Keep only the keys WorldModelTeacher.preprocess / the heads actually use.
    # Your episodes also store is_first/is_last/is_terminal/success/logprob,
    # which the world model does not consume; dropping them avoids stacking
    # bool/aux arrays through preprocess unnecessarily.
    keep = [k for k in ('image', 'action', 'reward', 'discount') if k in batch[0]]
    data = {}
    for key in keep:
        data[key] = np.stack([b[key] for b in batch], 0)
    return data


# --------------------------------------------------------------------------
# Core: run one task batch through encoder + dynamics, return the RSSM feature
#   feat = get_feat(post) = cat([stoch, deter])  -> shape [B, T, d]
# Using observe() with sample=False-equivalent (posterior mode via the same
# obs_step the training loop uses).
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_feat(wm, config, data_np, label):
    """Return per-timestep RSSM features for one task batch.

    Returns: feat  np.ndarray [B*T, d]
    """
    data = wm.preprocess(data_np)
    embed = wm.encoder(data, label=label)
    post, _prior = wm.dynamics.observe(embed, data['action'], label=label)
    feat = wm.dynamics.get_feat(post)            # [B, T, d]
    feat = feat.reshape(-1, feat.shape[-1])      # [B*T, d]
    return feat.detach().cpu().float().numpy()


@torch.no_grad()
def collect_expert_outputs(wm, config, data_np, label):
    """For the MoE encoder only: return stacked per-expert features.

    Returns: experts np.ndarray [n_experts, B*T, F]  (or None if not MoE)
    """
    if not hasattr(wm.encoder, 'experts'):
        return None
    data = wm.preprocess(data_np)
    _out, info = wm.encoder(data, label=label, return_expert_outputs=True)
    feats = info['expert_outputs']               # list of [B, T, F]
    stacked = []
    for ef in feats:
        ef = ef.reshape(-1, ef.shape[-1])        # [B*T, F]
        stacked.append(ef.detach().cpu().float().numpy())
    return np.stack(stacked, 0)                  # [N, B*T, F]

import argparse
import pathlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import analysis_common as ac


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
    p.add_argument('--n_episodes', type=int, default=8,
                   help='trajectory windows per task (batch size)')
    p.add_argument('--batch_length', type=int, default=50)
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _scatter(ax, X2d, labels, name_a, name_b, title):
    ax.scatter(X2d[labels == 0, 0], X2d[labels == 0, 1],
               c='#1f77b4', label=name_a, alpha=0.5, s=10)
    ax.scatter(X2d[labels == 1, 0], X2d[labels == 1, 1],
               c='#d62728', label=name_b, alpha=0.5, s=10)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overrides = {
        'teacher_encoder_mode': args.teacher_encoder_mode,
        'device': args.device,
        'teacher_model_path': args.teacher_model_path,
    }
    config = ac.load_config(args.configs, overrides)
    config.device = str(args.device)
    tools_seed = args.seed
    torch.manual_seed(tools_seed); np.random.seed(tools_seed)

    acts = ac.get_action_space(config)
    wm = ac.build_teacher(config, args.teacher_model_path, acts)

    label_a = ac.task_label(config, args.task_a)
    label_b = ac.task_label(config, args.task_b)
    print(f'[latent] {args.task_a} -> label {label_a}; {args.task_b} -> label {label_b}')

    data_a = ac.load_task_batch(config, args.task_a_dir,
                                args.n_episodes, args.batch_length, seed=args.seed)
    data_b = ac.load_task_batch(config, args.task_b_dir,
                                args.n_episodes, args.batch_length, seed=args.seed + 1)

    feat_a = ac.collect_feat(wm, config, data_a, label_a)   # [Na, d]
    feat_b = ac.collect_feat(wm, config, data_b, label_b)   # [Nb, d]

    latents = np.concatenate([feat_a, feat_b], 0)
    labels = np.concatenate([np.zeros(len(feat_a)), np.ones(len(feat_b))]).astype(int)
    print(f'[latent] collected {latents.shape[0]} timesteps, d={latents.shape[1]}')

    # Save raw for re-plotting / overlay later
    np.savez(outdir / f'{args.tag}_latents.npz',
             latents=latents, labels=labels,
             task_a=args.task_a, task_b=args.task_b)

    X_scaled = StandardScaler().fit_transform(latents)

    # ---- PCA ----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    var = pca.explained_variance_ratio_
    _scatter(ax, X_pca, labels, args.task_a, args.task_b,
             f'{args.tag} — PCA latent projection\n'
             f'(PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(outdir / f'{args.tag}_pca.png', dpi=160)
    plt.close(fig)

    # ---- Save Metrics to JSON ----
    json_path = outdir / f'{args.tag}_{args.task_a}_vs_{args.task_b}_projection.json'
    results_dict = {
        "tag": args.tag,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "timesteps": int(latents.shape[0]),
        "feature_dim": int(latents.shape[1]),
        "pca_explained_variance_ratio": var.tolist()
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f'[latent] wrote readable metrics to {json_path.name}')

    # ---- t-SNE ----
    perp = min(args.perplexity, max(5, (len(latents) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', random_state=args.seed)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _scatter(ax, X_tsne, labels, args.task_a, args.task_b,
             f'{args.tag} — t-SNE latent projection (perplexity={perp:.0f})')
    fig.tight_layout()
    fig.savefig(outdir / f'{args.tag}_tsne.png', dpi=160)
    plt.close(fig)

    # ---- MoE-only: per-expert feature PCA (shows orthogonal experts) ----
    experts_a = ac.collect_expert_outputs(wm, config, data_a, label_a)
    if experts_a is not None:
        N = experts_a.shape[0]
        stacked = experts_a.reshape(N * experts_a.shape[1], experts_a.shape[2])
        exp_labels = np.repeat(np.arange(N), experts_a.shape[1])
        Xe = StandardScaler().fit_transform(stacked)
        Xe2 = PCA(n_components=2).fit_transform(Xe)
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap('tab10')
        for e in range(N):
            m = exp_labels == e
            ax.scatter(Xe2[m, 0], Xe2[m, 1], s=8, alpha=0.5,
                       color=cmap(e % 10), label=f'expert {e}')
        ax.set_title(f'{args.tag} — per-expert feature space ({args.task_a})')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(outdir / f'{args.tag}_expert_pca.png', dpi=160)
        plt.close(fig)
        print(f'[latent] wrote {args.tag}_expert_pca.png')

    print(f'[latent] done. Plots in {outdir}/')


if __name__ == '__main__':
    main()
import argparse
import pathlib
import json

import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import analysis_common as ac


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--teacher_model_path', required=True)
    p.add_argument('--teacher_encoder_mode', default='moe',
                   choices=['moe', 'original_cnn'])
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
    p.add_argument('--pca_dims', type=int, default=0,
                   help='if >0, PCA-reduce features to this many dims before '
                        'silhouette (denoising). 0 = use raw features.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0],
                   help='one run per seed; reports mean +/- std across seeds')
    return p.parse_args()


def mahalanobis_centroid(Xa, Xb):
    """Centroid distance scaled by pooled within-class covariance.

    Consistent in spirit with the paper's mixScore (Eq. 5): larger = the two
    task clusters are further apart relative to their internal spread.
    """
    mu_a = Xa.mean(0)
    mu_b = Xb.mean(0)
    # pooled covariance of the two clusters (regularized for invertibility)
    Xa_c = Xa - mu_a
    Xb_c = Xb - mu_b
    n = len(Xa) + len(Xb)
    cov = (Xa_c.T @ Xa_c + Xb_c.T @ Xb_c) / max(n - 2, 1)
    cov += 1e-3 * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    diff = (mu_a - mu_b)
    return float(np.sqrt(diff @ inv @ diff))


def separability_metrics(feat_a, feat_b, pca_dims=0, seed=0):
    """Compute silhouette, linear-probe accuracy, Mahalanobis centroid dist."""
    X = np.concatenate([feat_a, feat_b], 0).astype(np.float64)
    y = np.concatenate([np.zeros(len(feat_a)), np.ones(len(feat_b))]).astype(int)

    # standardize: silhouette is distance-based, so unit-scale the features so
    # no single dim dominates purely by magnitude.
    X = StandardScaler().fit_transform(X)

    if pca_dims and pca_dims > 0 and pca_dims < X.shape[1]:
        # legitimate denoising on the REAL feature space (not a 2D viz)
        X = PCA(n_components=pca_dims, random_state=seed).fit_transform(X)

    sil = float(silhouette_score(X, y))  # on full/denoised features

    # linear probe: can a simple classifier separate the tasks?
    clf = LogisticRegression(max_iter=2000)
    acc = float(cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean())

    maha = mahalanobis_centroid(X[y == 0], X[y == 1])

    return {'silhouette': sil, 'probe_acc': acc, 'maha': maha}


def main():
    args = parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overrides = {
        'teacher_encoder_mode': args.teacher_encoder_mode,
        'device': args.device,
        'teacher_model_path': args.teacher_model_path,
        'precision': 32,   # clean float32 features, matches gradient_conflict.py
    }
    config = ac.load_config(args.configs, overrides)
    config.device = str(args.device)

    acts = ac.get_action_space(config)
    wm = ac.build_teacher(config, args.teacher_model_path, acts)

    label_a = ac.task_label(config, args.task_a)
    label_b = ac.task_label(config, args.task_b)
    print(f'[sep] {args.task_a} -> label {label_a}; '
          f'{args.task_b} -> label {label_b}')

    per_seed = {'silhouette': [], 'probe_acc': [], 'maha': []}
    for s in args.seeds:
        import torch
        torch.manual_seed(s)
        np.random.seed(s)

        data_a = ac.load_task_batch(config, args.task_a_dir,
                                    args.n_episodes, args.batch_length, seed=s)
        data_b = ac.load_task_batch(config, args.task_b_dir,
                                    args.n_episodes, args.batch_length,
                                    seed=s + 1000)

        feat_a = ac.collect_feat(wm, config, data_a, label_a)   # [Na, d]
        feat_b = ac.collect_feat(wm, config, data_b, label_b)   # [Nb, d]

        m = separability_metrics(feat_a, feat_b,
                                 pca_dims=args.pca_dims, seed=s)
        for k, v in m.items():
            per_seed[k].append(v)
        print(f'[sep] seed {s}: silhouette={m["silhouette"]:+.4f}  '
              f'probe_acc={m["probe_acc"]:.4f}  maha={m["maha"]:.4f}')

    # summarize
    print('\n==== SUMMARY ({} | {} vs {}) over {} seed(s) ===='.format(
        args.tag, args.task_a, args.task_b, len(args.seeds)))
    summary = {}
    for k, vals in per_seed.items():
        arr = np.array(vals)
        summary[k + '_mean'] = float(arr.mean())
        summary[k + '_std'] = float(arr.std())
        print(f'  {k:12s} = {arr.mean():+.4f} +/- {arr.std():.4f}')

    # Save to npz
    npz_path = outdir / f'{args.tag}_{args.task_a}_vs_{args.task_b}_separability.npz'
    np.savez(npz_path,
             tag=args.tag, task_a=args.task_a, task_b=args.task_b,
             seeds=np.array(args.seeds),
             silhouette=np.array(per_seed['silhouette']),
             probe_acc=np.array(per_seed['probe_acc']),
             maha=np.array(per_seed['maha']),
             **summary)
    
    # Save human-readable results to JSON
    json_path = outdir / f'{args.tag}_{args.task_a}_vs_{args.task_b}_separability.json'
    results_dict = {
        "tag": args.tag,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "seeds": args.seeds,
        "per_seed_results": per_seed,
        "summary": summary
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f'[sep] wrote arrays to {npz_path.name}')
    print(f'[sep] wrote readable values to {json_path.name}')


if __name__ == '__main__':
    main()
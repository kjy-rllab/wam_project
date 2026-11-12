"""Linear probes on the dumps of probe_features.py: what can be read from the residual stream, layer by layer.

Ridge regression from the hidden state of an image token (standardized) to what is true at its image cell, 5 folds
split by task so test kitchens and objects are from other tasks. Regression targets are scored by R^2 (log depth, xyz in
the base frame, distance to the gripper), binary ones by AUROC (cell on an object, on the robot, on the gripper, on
the task's object, on a fixture the task refers to; and, among cells on objects, task object against distractors).
From the last prompt token: gripper position, and the task
object's position relative to the gripper. Baselines: the cell's own pixels with its position (pixels), and the same
probes on shuffled scenes (shuffled: labels of another scene's tokens, which keeps each label's statistics).

    python -m qwen_vla.probe_linear --features <dir> --output <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PENALTIES = (1e-2, 1e-1, 1.0, 10.0)  # ridge penalty in units of the mean eigenvalue of X^T X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paths = sorted(args.features.glob('*.npz'))
    data = [np.load(p) for p in paths]
    tasks = sorted({p.stem.rsplit('_', 1)[0] for p in paths})
    fold = lambda p: tasks.index(p.stem.rsplit('_', 1)[0]) % args.folds
    counts = [len(d['depth']) for d in data]
    token_fold = torch.tensor(np.concatenate([np.full(n, fold(p)) for p, n in zip(paths, counts)]), device=device)
    scene_fold = torch.tensor([fold(p) for p in paths], device=device)
    cat = lambda key: np.concatenate([d[key] for d in data])

    cover, kind = cat('cover'), cat('kind')
    token_targets = {
        'log_depth': np.log(cat('depth'))[:, None], 'xyz': cat('xyz'), 'reach': cat('reach')[:, None],
        'on_object': (kind == 2)[:, None], 'on_robot': (kind == 3)[:, None], 'on_gripper': cover[:, 2:3] > 0.2,
        'on_task_object': cover[:, 0:1] > 0.2, 'on_task_fixture': cover[:, 1:2] > 0.2,
        # among cells on objects only: is it the task's object or a distractor? (needs the instruction)
        'on_task_object_among_objects': np.where(kind == 2, cover[:, 0] > 0.2, np.nan)[:, None],
    }
    scene_targets = {'eef': np.stack([d['eef'] for d in data]), 'object_from_gripper': np.stack([d['object'] for d in data])}
    binary = {k for k in token_targets if k.startswith('on_')}
    position = np.stack([cat('row'), cat('col')], axis=1)
    position = np.concatenate([position / position.max(), np.eye(3)[cat('camera').astype(int)]], axis=1)
    order = np.random.default_rng(0).permutation(len(data))  # shuffled baseline: every scene gets another scene's labels
    shuffled = lambda key: np.concatenate([data[j][key] for j in order])  # scenes have equally many tokens

    def probe(features: torch.Tensor, targets: dict, folds: torch.Tensor) -> dict:
        """Cross-validated ridge. The penalty is chosen on one training fold."""
        scores = {}
        for name, target in targets.items():
            y = torch.tensor(np.asarray(target, dtype=np.float32), device=device)
            valid = ~y.isnan().any(1)
            y = y.nan_to_num()
            predictions = torch.zeros_like(y)
            for f in range(args.folds):
                train, test, inner = valid & (folds != f), valid & (folds == f), valid & (folds == (f + 1) % args.folds)
                fit = train & ~inner
                mean, std = features[fit].mean(0), features[fit].std(0) + 1e-6
                x = (features - mean) / std
                best = None
                for rows, held in ((fit, inner), (train, test)):
                    values, vectors = torch.linalg.eigh(x[rows].T @ x[rows])
                    projected, offset = vectors.T @ (x[rows].T @ (y[rows] - y[rows].mean(0))), y[rows].mean(0)
                    penalties = PENALTIES if best is None else (best,)
                    errors = []
                    for penalty in penalties:
                        weights = vectors @ (projected / (values[:, None] + penalty * values.mean()))
                        errors.append(((x[held] @ weights + offset - y[held]) ** 2).mean().item())
                    best = penalties[int(np.argmin(errors))]
                predictions[test] = x[test] @ weights + offset
            y, predictions = y[valid], predictions[valid]
            if name in binary:
                ranks = predictions[:, 0].argsort().argsort().float() + 1
                positive = y[:, 0] > 0.5
                scores[name] = ((ranks[positive].mean() - (positive.sum() + 1) / 2) / (~positive).sum()).item()
            else:
                scores[name] = (1 - ((predictions - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()).item()
        return scores

    layers = data[0]['hidden'].shape[0]
    results = {'layers': layers, 'scenes': len(data), 'tokens': int(sum(counts)), 'token': [], 'last': [], 'shuffled': []}
    pixels = torch.tensor(np.concatenate([cat('pixels') / 255, position], axis=1), dtype=torch.float32, device=device)
    results['pixels'] = probe(pixels, token_targets, token_fold)
    results['position'] = probe(pixels[:, 48:], token_targets, token_fold)
    for layer in tqdm(range(layers)):
        hidden = torch.tensor(np.concatenate([d['hidden'][layer, :-1] for d in data]), dtype=torch.float32, device=device)
        results['token'].append(probe(hidden, token_targets, token_fold))
        results['shuffled'].append(probe(hidden, {'log_depth': np.log(shuffled('depth'))[:, None],
                                                  'on_object': (shuffled('kind') == 2)[:, None]}, token_fold))
        last = torch.tensor(np.stack([d['hidden'][layer, -1] for d in data]), dtype=torch.float32, device=device)
        results['last'].append(probe(last, scene_targets, scene_fold))
    args.output.write_text(json.dumps(results, indent=1))
    for name in list(token_targets) + list(scene_targets):
        rows = results['token'] if name in token_targets else results['last']
        base = f'pixels {results["pixels"][name]:.2f} position {results["position"][name]:.2f} | ' if name in token_targets else ''
        print(f'{name:20} {base}' + ' '.join(f'{r[name]:.2f}' for r in rows))


if __name__ == '__main__':
    main()

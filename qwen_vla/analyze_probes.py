"""Figures and tables from one model's probe outputs (probe_behavior, probe_attention, probe_linear).

    python -m qwen_vla.analyze_probes <probe output dir> <results dir>

Writes <results dir>/summary.json and figures/: behavior.png, attention_layers.png, attention_heads.png,
attention_examples.png, probes.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE, ORANGE, AQUA, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'grid.color': '#e4e3df', 'axes.grid': True, 'grid.linewidth': 0.6,
                     'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 140})


def wilson(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p, z = k / n, 1.96
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main() -> None:
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    summary = {}

    # Behavior: grounding by what is asked for, relations by kind, per image size
    sizes = sorted({int(p.stem.split('_')[1]) for p in source.glob('behavior_*.jsonl')})
    if sizes:
        rows = {s: [json.loads(line) for p in sorted(source.glob(f'behavior_{s}_*.jsonl')) for line in open(p)] for s in sizes}
        groups = [('object', 'ground', 'role'), ('appliance', 'ground', 'role'), ('gripper', 'ground', 'role'),
                  ('left', 'left', 'kind'), ('camera', 'camera', 'kind'), ('gripper', 'gripper', 'kind')]
        labels = ['locate\nobject', 'locate\nappliance', 'locate\ngripper', 'further\nleft', 'closer to\ncamera', 'closer to\ngripper']
        fig, ax = plt.subplots(figsize=(7.2, 3.0), layout='constrained')
        summary['behavior'] = {}
        for i, size in enumerate(sizes):
            values, low, high = [], [], []
            for (name, kind, field), label in zip(groups, labels):
                part = [r for r in rows[size] if r['kind'] == kind and (field == 'kind' or r['role'] == name)]
                hits = sum((r['iou'] >= 0.5) if kind == 'ground' else r['correct'] for r in part)
                interval = wilson(hits, len(part))
                values.append(hits / max(len(part), 1)); low.append(values[-1] - interval[0]); high.append(interval[1] - values[-1])
                summary['behavior'][f'{size}/{label.replace(chr(10), " ")}'] = {
                    'n': len(part), 'rate': values[-1], 'ci': interval,
                    **({'mean_iou': float(np.mean([r['iou'] for r in part])), 'centre_on_target': float(np.mean([r['hit'] for r in part]))} if kind == 'ground' and part else {})}
            x = np.arange(len(groups)) + (i - (len(sizes) - 1) / 2) * 0.8 / len(sizes)
            ax.bar(x, values, 0.8 / len(sizes) - 0.03, yerr=[low, high], color=[BLUE, ORANGE, AQUA, MUTED][i], error_kw={'lw': 0.8, 'ecolor': INK}, label=f'{size} px')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.text(5.45, 0.51, 'chance (relations)', color=MUTED, fontsize=7, ha='right')
        ax.set_xticks(np.arange(len(groups)), labels)
        ax.set_ylim(0, 1); ax.set_ylabel('IoU ≥ 0.5 (locate) / correct (relations)'); ax.legend(ncol=len(sizes), loc='upper right')
        fig.savefig(out / 'figures/behavior.png'); plt.close(fig)

    # Attention: how well each layer's read of the image tokens singles out a mask
    for path in sorted(source.glob('attention_*_0.npz')):
        tag = path.stem.rsplit('_', 1)[0]  # attention_<mode>_<size>
        parts = [np.load(p, allow_pickle=True) for p in sorted(source.glob(f'{tag}_*.npz'))]
        auroc, point, share = (np.concatenate([p[k] for p in parts]) for k in ('auroc', 'point', 'image_share'))
        masks, types = [str(m) for m in parts[0]['masks']], np.array([str(t) for t in parts[0]['types']])
        layers, full = np.arange(len(types)), types == 'full_attention'
        queries = ['phrase token', 'last prompt token', 'box token'] if 'locate' in tag else ['instruction end', 'last prompt token']
        mean = np.nanmean(auroc, 0)  # [masks, queries, layers, heads + 1]
        summary[tag] = {'examples': int(len(auroc)), 'masks': masks, 'queries': queries,
                        'auroc_all_heads': {m: {q: mean[i, j, :, -1].tolist() for j, q in enumerate(queries)} for i, m in enumerate(masks)},
                        'best_head_auroc': {q: np.nanmax(mean[0, j, :, :-1], -1).tolist() for j, q in enumerate(queries)},
                        'pointing_all_heads': {q: np.nanmean(point[:, 0, j, :, -1], 0).tolist() for j, q in enumerate(queries)},
                        'image_share': {q: np.nanmean(share[:, j], 0).tolist() for j, q in enumerate(queries)}}
        fig, axes = plt.subplots(2, len(queries), figsize=(3.6 * len(queries), 4.6), layout='constrained', sharex=True, sharey='row', squeeze=False)
        for j, query in enumerate(queries):
            for i, (mask, color) in enumerate(zip(masks, (BLUE, ORANGE, AQUA))):
                axes[0, j].plot(layers, mean[i, j, :, -1], color=color, lw=1.0, label=mask)
                axes[0, j].scatter(layers[full], mean[i, j, full, -1], color=color, s=14, zorder=3)
            axes[0, j].plot(layers, np.nanmax(mean[0, j, :, :-1], -1), color=BLUE, lw=0.8, ls='--', label=f'{masks[0]}, best head')
            axes[0, j].axhline(0.5, color=MUTED, lw=0.8, ls=':')
            axes[0, j].set_title(f'read from the {query}', fontsize=9)
            axes[1, j].plot(layers, np.nanmean(share[:, j], 0), color=INK, lw=1.0)
            axes[1, j].scatter(layers[full], np.nanmean(share[:, j], 0)[full], color=INK, s=14, zorder=3)
        axes[0, 0].set_ylabel('AUROC of mask cells'); axes[1, 0].set_ylabel('share of the write from image tokens')
        axes[0, 0].legend(fontsize=7)
        fig.supxlabel('layer (dots: softmax attention layers; the others are Gated DeltaNet layers)', fontsize=9, color=INK)
        fig.savefig(out / f'figures/{tag}_layers.png'); plt.close(fig)

        # Heads: AUROC of the first mask per layer and head, for the query that reads the image most
        q = int(np.argmax([np.nanmax(mean[0, j, :, -1]) for j in range(len(queries))]))
        fig, ax = plt.subplots(figsize=(7.2, 3.2), layout='constrained')
        image = ax.imshow(mean[0, q, :, :-1].T, aspect='auto', cmap='Blues', vmin=0.5, vmax=1.0, interpolation='nearest')
        ax.grid(False); ax.set_xlabel('layer'); ax.set_ylabel('head'); ax.set_title(f'AUROC of the {masks[0]} per head, read from the {queries[q]}', fontsize=9)
        fig.colorbar(image, ax=ax, pad=0.01)
        fig.savefig(out / f'figures/{tag}_heads.png'); plt.close(fig)

    # Linear probes: one panel per target, lines per dump
    dumps = {p.stem.removeprefix('probe_'): json.loads(p.read_text()) for p in sorted(source.glob('probe_*.json'))}
    if dumps:
        summary['probes'] = dumps
        targets = list(next(iter(dumps.values()))['token'][0])
        fig, axes = plt.subplots(2, (len(targets) + 1) // 2, figsize=(2.6 * ((len(targets) + 1) // 2), 4.8), layout='constrained', squeeze=False)
        for ax, target in zip(axes.flat, targets):
            for (name, dump), color in zip(dumps.items(), (BLUE, ORANGE, AQUA)):
                x = np.linspace(0, 1, dump['layers'])
                ax.plot(x, [r[target] for r in dump['token']], color=color, label=name)
                ax.axhline(dump['pixels'][target], color=color, lw=0.8, ls=':')
            ax.set_title(target.replace('_', ' '), fontsize=8)
            ax.set_ylim(0.4 if target.startswith('on_') else -0.05, 1.0)
        for ax in axes[1]:
            ax.set_xlabel('depth (0: vision encoder output, 1: last layer)')
        axes[0, 0].set_ylabel('AUROC (on …) / R² (others)'); axes[0, 0].legend(fontsize=7)
        fig.savefig(out / 'figures/probes.png'); plt.close(fig)

    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k == 'behavior'}, indent=1)[:3000])


if __name__ == '__main__':
    main()

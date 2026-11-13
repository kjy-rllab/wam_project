"""Figures and tables from one model's probe outputs (probe_behavior, probe_attention, probe_linear).

    python -m qwen_vla.analyze_probes <probe output dir> <results dir> <scenes dir>

Writes <results dir>/summary.json and figures/: behavior.png, probes.png and, per attention run,
attention_<mode>_<size>_layers.png, _heads.png and _examples.png.
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
COLORS = (BLUE, ORANGE, AQUA, MUTED)
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'grid.color': '#e4e3df', 'axes.grid': True, 'grid.linewidth': 0.6,
                     'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 140})
# Names in the figures. Query positions: the token whose incoming contributions are scored
QUERIES = {'locate': ['referring expression, last token', 'final input token', 'x1 prediction position'],
           'policy': ['instruction, last token', 'final input token']}
MASKS = {'target': 'queried entity', 'control': 'non-queried entity', 'object': 'instructed object', 'distractor': 'other objects',
         'fixture': 'task fixture', 'gripper': 'gripper'}
TARGETS = {'log_depth': 'log depth (R²)', 'xyz': '3D position, base frame (R²)', 'reach': 'distance to end-effector (R²)',
           'on_object': 'object patch (AUROC)', 'on_robot': 'robot patch (AUROC)', 'on_gripper': 'gripper patch (AUROC)',
           'on_task_object': 'task-object patch (AUROC)', 'on_task_fixture': 'task-fixture patch (AUROC)'}
LAYOUTS = {'policy': '3 views x 256 px, image-first', 'policy_text_first': '3 views x 256 px, text-first', 'single_768': '1 view x 768 px'}


def wilson(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p, z = k / n, 1.96
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main() -> None:
    source, out, scene_dir = (Path(a) for a in sys.argv[1:4])
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    summary = {}

    # Behavior: grounding by what is asked for, relations by kind, per image size
    sizes = sorted({int(p.stem.split('_')[1]) for p in source.glob('behavior_*.jsonl')})
    if sizes:
        rows = {s: [json.loads(line) for p in sorted(source.glob(f'behavior_{s}_*.jsonl')) for line in open(p)] for s in sizes}
        groups = [('object', 'ground', 'role'), ('appliance', 'ground', 'role'), ('gripper', 'ground', 'role'),
                  ('left', 'left', 'kind'), ('camera', 'camera', 'kind'), ('gripper', 'gripper', 'kind')]
        labels = ['grounding\nobject', 'grounding\nappliance', 'grounding\ngripper', 'relation\nleft-right', 'relation\ndepth order', 'relation\ngripper distance']
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
            ax.bar(x, values, 0.8 / len(sizes) - 0.03, yerr=[low, high], color=COLORS[i], error_kw={'lw': 0.8, 'ecolor': INK},
                   label=f'{size} px' + (' (256 px upsampled)' if size in (384, 512) else ''))
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.text(2.42, 0.515, 'chance level', color=MUTED, fontsize=7, ha='right')
        ax.set_xticks(np.arange(len(groups)), labels)
        ax.set_ylim(0, 1.12); ax.set_ylabel('Acc@IoU≥0.5 / accuracy'); ax.legend(ncol=len(sizes), loc='upper right', fontsize=7)
        fig.savefig(out / 'figures/behavior.png'); plt.close(fig)

    # Attention: how well each layer's contribution map over the image tokens localizes a mask. Layers are numbered from 1
    depth = None
    for path in sorted(source.glob('attention_*_0.npz')):
        tag = path.stem.rsplit('_', 1)[0]  # attention_<mode>_<size>
        parts = [np.load(p, allow_pickle=True) for p in sorted(source.glob(f'{tag}_*.npz'))]
        auroc, point, share = (np.concatenate([p[k] for p in parts]) for k in ('auroc', 'point', 'image_share'))
        masks, types = [str(m) for m in parts[0]['masks']], np.array([str(t) for t in parts[0]['types']])
        layers, full = np.arange(len(types)), types == 'full_attention'
        queries, depth = QUERIES['locate' if 'locate' in tag else 'policy'], len(types)
        mean = np.nanmean(auroc, 0)  # [masks, queries, layers, heads + 1]
        summary[tag] = {'examples': int(len(auroc)), 'with_mask': np.isfinite(auroc[:, :, 0, 0, -1]).sum(0).tolist(), 'masks': masks, 'queries': queries,
                        'auroc_all_heads': {m: {q: mean[i, j, :, -1].tolist() for j, q in enumerate(queries)} for i, m in enumerate(masks)},
                        'best_head_auroc': {q: np.nanmax(mean[0, j, :, :-1], -1).tolist() for j, q in enumerate(queries)},
                        'pointing_all_heads': {q: np.nanmean(point[:, 0, j, :, -1], 0).tolist() for j, q in enumerate(queries)},
                        'image_share': {q: np.nanmean(share[:, j], 0).tolist() for j, q in enumerate(queries)}}
        fig, axes = plt.subplots(2, len(queries), figsize=(3.6 * len(queries), 4.6), layout='constrained', sharex=True, sharey='row', squeeze=False)
        for j, query in enumerate(queries):
            for i, (mask, color) in enumerate(zip(masks, COLORS)):
                axes[0, j].plot(layers + 1, mean[i, j, :, -1], color=color, lw=1.0, label=MASKS[mask])
                axes[0, j].scatter(layers[full] + 1, mean[i, j, full, -1], color=color, s=14, zorder=3)
            axes[0, j].plot(layers + 1, np.nanmax(mean[0, j, :, :-1], -1), color=BLUE, lw=0.8, ls='--', label=f'{MASKS[masks[0]]}, best single head')
            axes[0, j].axhline(0.5, color=MUTED, lw=0.8, ls=':')
            axes[0, j].set_title(f'query position: {query}', fontsize=9)
            axes[1, j].plot(layers + 1, np.nanmean(share[:, j], 0), color=INK, lw=1.0)
            axes[1, j].scatter(layers[full] + 1, np.nanmean(share[:, j], 0)[full], color=INK, s=14, zorder=3)
        axes[0, 0].set_ylabel('localization AUROC'); axes[1, 0].set_ylabel('image-token share')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=5, fontsize=7)
        fig.supxlabel('layer (markers: softmax attention layers; the others are Gated DeltaNet layers)', fontsize=9, color=INK)
        fig.savefig(out / f'figures/{tag}_layers.png'); plt.close(fig)

        # Heads: AUROC of the first mask per layer and head, for the query position where it is highest
        q = int(np.argmax([np.nanmax(mean[0, j, :, -1]) for j in range(len(queries))]))
        fig, ax = plt.subplots(figsize=(7.2, 3.2), layout='constrained')
        image = ax.imshow(mean[0, q, :, :-1].T, aspect='auto', cmap='Blues', vmin=0.5, vmax=1.0, interpolation='nearest',
                          extent=(0.5, len(types) + 0.5, mean.shape[-1] - 1.5, -0.5))
        ax.grid(False); ax.set_xlabel('layer'); ax.set_ylabel('head'); ax.set_title(f'localization AUROC of the {MASKS[masks[0]]} per head; query position: {queries[q]}', fontsize=9)
        fig.colorbar(image, ax=ax, pad=0.01)
        fig.savefig(out / f'figures/{tag}_heads.png'); plt.close(fig)

        # Examples: the contribution map at the query position, layer by layer (first mask outlined)
        meta = json.loads(str(parts[0]['meta']))
        shown = [e for e in range(len(meta)) if f'maps_{e}' in parts[0] and parts[0][f'cover_{e}'][0].max() >= 0.2][:4]
        picked = np.concatenate([layers[full][np.linspace(0, full.sum() - 1, 6).round().astype(int)], layers[full][[full.sum() // 2]] - 1])
        picked = np.sort(picked)
        wide = 'camera' not in meta[0]  # policy: the three cameras side by side
        width, height = (1.95, 0.9) if wide else (1.3, 1.5)
        fig, axes = plt.subplots(len(shown), 1 + len(picked), figsize=(width * (1 + len(picked)), height * len(shown) + 0.3), layout='constrained', squeeze=False)
        for row, e in zip(axes, shown):
            scene = np.load(scene_dir / f'{meta[e]["scene"]}.npz')
            picture = np.concatenate([scene['rgb_256'][c] for c in ([0, 1, 2] if wide else [['left', 'right', 'wrist'].index(meta[e]['camera'])])], axis=1)
            grids, maps, cover = parts[0][f'grids_{e}'], parts[0][f'maps_{e}'].astype(np.float32)[:, q, -1], parts[0][f'cover_{e}'][0]
            tile = lambda v: np.concatenate([g.reshape(shape) for g, shape in zip(np.split(v, np.cumsum(grids.prod(1))[:-1]), grids)], axis=1)
            extent = (0, picture.shape[1], picture.shape[0], 0)
            row[0].imshow(picture)
            row[0].set_title((meta[e].get('phrase') or meta[e]['scene'])[:30], fontsize=7)
            for ax, layer in zip(row[1:], picked):
                heat = tile(maps[layer] / maps[layer].max())
                ax.imshow(255 - (255 - picture.mean(-1)) * 0.35, cmap='gray', vmin=0, vmax=255)
                ax.imshow(np.dstack([np.full(heat.shape + (3,), matplotlib.colors.to_rgb(BLUE)), heat**0.7]), extent=extent, interpolation='nearest')
                ax.set_title(f'layer {layer + 1}' + ('' if full[layer] else ' (DeltaNet)'), fontsize=7)
            for ax in row:
                ax.contour(np.kron(tile(cover), np.ones((8, 8))), [0.2], colors=ORANGE, linewidths=0.8, extent=extent, origin='upper')
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.suptitle(f'contribution map of the image tokens at the query position ({queries[q]}); outline: {MASKS[masks[0]]} mask', fontsize=8)
        fig.savefig(out / f'figures/{tag}_examples.png'); plt.close(fig)

    # Linear probes: one panel per target, lines per dump
    dumps = {p.stem.removeprefix('probe_'): json.loads(p.read_text()) for p in sorted(source.glob('probe_*.json'))}
    if dumps:
        summary['probes'] = dumps
        fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.8), layout='constrained')
        for ax, (target, title) in zip(axes.flat, TARGETS.items()):
            for (name, dump), color in zip(dumps.items(), (BLUE, ORANGE, AQUA)):
                x = np.linspace(0, depth or 1, dump['layers'])  # hidden states from the embeddings to the last layer
                ax.plot(x, [r[target] for r in dump['token']], color=color, label=LAYOUTS.get(name, name))
                ax.axhline(dump['pixels'][target], color=color, lw=0.8, ls=':')
            ax.set_title(title, fontsize=8)
            ax.set_ylim(0.4 if target.startswith('on_') else -0.05, 1.0)
        fig.supxlabel(('layer (0: vision encoder output)' if depth else 'relative depth (0: vision encoder output, 1: last layer)')
                      + '; dotted: ridge baseline on the patch\'s pixels and position', fontsize=9, color=INK)
        fig.supylabel('cross-validated score of the linear probe', fontsize=9, color=INK); axes[0, 0].legend(fontsize=7)
        fig.savefig(out / 'figures/probes.png'); plt.close(fig)

    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k == 'behavior'}, indent=1)[:3000])


if __name__ == '__main__':
    main()

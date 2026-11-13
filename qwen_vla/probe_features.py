"""Dump the VLM's residual stream at image tokens of probe scenes, with what is true at each token's image cell.

policy: the three cameras at 256 and the task instruction in one sequence, as a policy would see them (--text-first
puts the instruction before the images, so image tokens can depend on it). single: one camera image at --size per
sequence. Per scene, <output>/<scene>.npz holds hidden [layers, tokens, D] (every --step-th layer of the residual
stream at the kept image tokens, then the last prompt token) and, per kept image token: camera, row, col, pixels (the
cell's colours, 4x4x3), depth (median, m; nan beyond 8 m), xyz (cell centre in the robot base frame, m), reach (distance to the
gripper, m), kind (majority entity: 0 none/other, 1 fixture, 2 object, 3 robot) and the cell's coverage by the task's
object, the fixtures the task refers to, and the gripper. Scene-level: eef (base frame), gripper_qpos, object (the
task object's position minus the gripper's, base frame; nan without one).

    python -m qwen_vla.probe_features --model <checkpoint> --scenes <scenes dir> --output <dir> --layout policy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qwen_vla import scenes, vlm

KINDS = {'none': 0, 'other': 0, 'fixture': 1, 'object': 2, 'robot': 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--weights', help='vlm.safetensors of a full-mode training checkpoint (the fine-tuned VLM)')
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout', choices=('policy', 'single'), default='policy')
    parser.add_argument('--text-first', action='store_true')
    parser.add_argument('--size', type=int, default=768, choices=(256, 768), help='image side (single)')
    parser.add_argument('--step', type=int, default=2, help='keep every N-th layer')
    parser.add_argument('--tokens', type=int, default=64, help='image tokens kept per image (random beyond that)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    args = parser.parse_args()

    model, processor = vlm.load(args.model, weights=args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    size = 256 if args.layout == 'policy' else args.size
    for path in tqdm(sorted(args.scenes.glob('*.npz'))[:args.limit][args.shard[0]::args.shard[1]]):
        scene = scenes.load(path)
        names = [e['name'] for e in scene['entities']]
        kinds = np.array([KINDS[e['kind']] for e in scene['entities']])
        frames = [Image.fromarray(f) for f in scene['rgb' if size == 768 else 'rgb_256']]
        prompts = [(frames, scene['lang'])] if args.layout == 'policy' else [([f], scene['lang']) for f in frames]
        rng = np.random.default_rng(0)

        # World -> robot base frame (quaternions are xyzw)
        x, y, z, w = scene['robot']['base_quat']
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        to_base = lambda p: (np.asarray(p) - scene['robot']['base_pos']) @ rotation
        eef = np.array(scene['robot']['eef_pos'])
        roles = [scene['mask'] == (names.index('obj') if 'obj' in names else -1),
                 np.isin(scene['mask'], [names.index(n) for n in scene['fixture_refs'].values() if n in names]),
                 scene['mask'] == (names.index('gripper') if 'gripper' in names else -1)]

        hidden, labels = [], {k: [] for k in ('camera', 'row', 'col', 'pixels', 'depth', 'xyz', 'reach', 'kind', 'cover')}
        for p, prompt in enumerate(prompts):
            inputs = vlm.chat(processor, [prompt], text_first=args.text_first).to(model.device)
            with torch.no_grad():
                states = torch.stack(model.model(**inputs, output_hidden_states=True).hidden_states)[::args.step, 0]
            last = [states[:, -1:]]
            for g, grid in enumerate(vlm.image_tokens(model, inputs)):
                camera, (rows, cols) = p + g, grid.shape
                keep = np.sort(rng.permutation(rows * cols)[:args.tokens])
                hidden.append(states[:, grid.flatten()[keep]])
                side = scene['mask'].shape[1]
                cell = side // rows
                depth = np.median(scene['depth'][camera].astype(np.float32).reshape(rows, cell, cols, cell).transpose(0, 2, 1, 3).reshape(rows * cols, -1), axis=1)
                v, u = np.divmod(np.arange(rows * cols), cols)
                K, E = scene['K'][camera], scene['E'][camera]
                depth = np.where(depth < 8, depth, np.nan)  # cells that look out of the kitchen have no depth
                points = np.stack([((u + 0.5) * cell - K[0, 2]) / K[0, 0] * depth, ((v + 0.5) * cell - K[1, 2]) / K[1, 1] * depth, depth, np.ones_like(depth)], axis=1) @ E.T
                majority = np.array([np.bincount(c, minlength=4).argmax() for c in kinds[scene['mask'][camera]].reshape(rows, cell, cols, cell).transpose(0, 2, 1, 3).reshape(rows * cols, -1)])
                pixels = np.asarray(frames[camera].resize((cols * 4, rows * 4), Image.BILINEAR)).reshape(rows, 4, cols, 4, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, 48)
                for key, value in (('camera', np.full(rows * cols, camera)), ('row', v), ('col', u), ('pixels', pixels), ('depth', depth),
                                   ('xyz', to_base(points[:, :3])), ('reach', np.linalg.norm(points[:, :3] - eef, axis=1)), ('kind', majority),
                                   ('cover', np.stack([scenes.cells(r[camera].astype(np.float32), rows, cols).ravel() for r in roles], axis=1))):
                    labels[key].append(value[keep])
        np.savez(
            args.output / f'{scene["path"]}.npz', hidden=torch.cat(hidden + last, dim=1).half().cpu().numpy(),
            **{k: np.concatenate(v).astype(np.float32) for k, v in labels.items()}, eef=to_base(eef),
            gripper_qpos=scene['robot']['gripper_qpos'],
            object=to_base(scene['objects']['obj']['pos']) - to_base(eef) if 'obj' in scene['objects'] else np.full(3, np.nan),
        )


if __name__ == '__main__':
    main()

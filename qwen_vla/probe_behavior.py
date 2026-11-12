"""Can the VLM say where things are? Grounding and spatial relations on RoboCasa probe scenes (collect_scenes.py).

Grounding: 'Locate the {phrase}' -> box, scored by IoU with the box of the ground-truth mask and by whether the box
centre lies on the thing. Relations between two things in an image, asked in both orders: which is further left, which
is closer to the camera (median depth of the mask), which is closer to the gripper (3D distance, objects only, gripper
in view). Pairs whose ground truth is nearly tied are skipped.

    python qwen_vla/probe_behavior.py --model <checkpoint> --scenes <scenes dir> --output <dir> [--size 256]

Writes behavior_<size>.jsonl (one line per question) and prints the summary.
"""
import argparse
import itertools
import json
import re
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qwen_vla import scenes, vlm

GROUND = 'Locate the {} in the image. Output its bounding box in JSON.'
RELATIONS = {
    'left': 'Which is further to the left in the image, the {} or the {}? Answer with only its name.',
    'camera': 'Which is closer to the camera, the {} or the {}? Answer with only its name.',
    'gripper': 'Which is closer to the robot gripper, the {} or the {}? Answer with only its name.',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=768, choices=(256, 384, 512, 768),
                        help='image side given to the model; 384 and 512 are the 256 render upsampled')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--pairs', type=int, default=2, help='pairs per relation and image, each asked in both orders')
    parser.add_argument('--limit', type=int, help='use only the first N scenes')
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    args = parser.parse_args()

    model, processor = vlm.load(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    out = open(args.output / f'behavior_{args.size}_{args.shard[0]}.jsonl', 'w')
    paths = sorted(args.scenes.glob('*.npz'))[:args.limit][args.shard[0]::args.shard[1]]
    for path in tqdm(paths):
        scene = scenes.load(path)
        eef = np.array(scene['robot']['eef_pos'])
        for camera in range(3):
            image = Image.fromarray(scene['rgb' if args.size == 768 else 'rgb_256'][camera]).resize((args.size, args.size), Image.BICUBIC)
            things = scenes.targets(scene, camera)
            rng = np.random.default_rng([zlib.crc32(scene['path'].encode()), camera])  # same pairs for every model
            items = [{'kind': 'ground', 'a': t['phrase'], 'role': t['role'], 'text': GROUND.format(t['phrase'])}
                     for t in things]
            # Relations: value is smaller for the answer; pairs that are nearly tied are skipped
            pairs = {kind: [] for kind in RELATIONS}
            for a, b in itertools.combinations(things, 2):
                centre = [(t['box'][0] + t['box'][2]) / 2 for t in (a, b)]
                depth = [float(np.median(scene['depth'][camera][t['mask']])) for t in (a, b)]
                if abs(centre[0] - centre[1]) >= 100:
                    pairs['left'].append((a, b, centre))
                if abs(np.log(depth[0] / depth[1])) >= 0.15:
                    pairs['camera'].append((a, b, depth))
                if a['pos'] and b['pos'] and 'robot gripper' in [t['phrase'] for t in things]:
                    reach = [float(np.linalg.norm(np.array(t['pos']) - eef)) for t in (a, b)]
                    if abs(np.log(reach[0] / reach[1])) >= 0.3:
                        pairs['gripper'].append((a, b, reach))
            for kind, found in pairs.items():
                for i in rng.permutation(len(found))[:args.pairs]:
                    a, b, value = found[i]
                    for x, y in ((a, b), (b, a)):  # both orders
                        items.append({'kind': kind, 'a': x['phrase'], 'b': y['phrase'], 'value': value,
                                      'answer': (a, b)[int(np.argmin(value))]['phrase'],
                                      'text': RELATIONS[kind].format(x['phrase'], y['phrase'])})
            for start in range(0, len(items), args.batch):
                batch = items[start:start + args.batch]
                inputs = vlm.chat(processor, [([image], item['text']) for item in batch]).to(model.device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
                replies = processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                for item, reply in zip(batch, replies):
                    item |= {'scene': scene['path'], 'camera': scenes.CAMERAS[camera], 'reply': reply.strip()}
                    if item['kind'] == 'ground':
                        t = next(t for t in things if t['phrase'] == item['a'])
                        numbers = re.search(r'\[\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\]', reply)
                        item |= {'box': t['box'].tolist(), 'area': float(t['mask'].mean()), 'iou': 0.0, 'hit': False}
                        if numbers:
                            x1, y1, x2, y2 = pred = [float(n) for n in numbers.groups()]
                            gx1, gy1, gx2, gy2 = t['box']
                            inter = max(0, min(x2, gx2) - max(x1, gx1)) * max(0, min(y2, gy2) - max(y1, gy1))
                            union = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
                            side = t['mask'].shape[0]
                            cy, cx = (min(side - 1, int(v * side / 1000)) for v in ((y1 + y2) / 2, (x1 + x2) / 2))
                            item |= {'pred': pred, 'iou': inter / max(union, 1e-9), 'hit': bool(t['mask'][cy, cx])}
                    else:
                        said = [p for p in (item['a'], item['b']) if p in reply.lower()]
                        item |= {'said': said[0] if len(said) == 1 else None}
                        item['correct'] = item['said'] == item['answer']
                    out.write(json.dumps(item) + '\n')
            out.flush()
    out.close()

    rows = [json.loads(line) for line in open(out.name)]
    ground = [r for r in rows if r['kind'] == 'ground']
    for role in ('object', 'appliance', 'gripper'):
        part = [r for r in ground if r['role'] == role]
        if part:
            print(f'ground {role:9} n={len(part):4}  IoU>=0.5 {np.mean([r["iou"] >= 0.5 for r in part]):.3f}  '
                  f'mean IoU {np.mean([r["iou"] for r in part]):.3f}  centre on target {np.mean([r["hit"] for r in part]):.3f}')
    for kind in RELATIONS:
        part = [r for r in rows if r['kind'] == kind]
        if part:
            print(f'{kind:8} n={len(part):4}  correct {np.mean([r["correct"] for r in part]):.3f}  '
                  f'no answer {np.mean([r["said"] is None for r in part]):.3f}')


if __name__ == '__main__':
    main()

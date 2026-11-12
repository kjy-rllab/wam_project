"""Where does the VLM look? What every layer reads from the image tokens, scored against ground-truth masks.

locate: one camera image and 'Locate the {phrase} ...' for a visible thing. policy: the three cameras at 256 and the
task instruction, as a policy would see them. vlm.trace gives, for every layer and head, how much each image token adds
to the residual stream at three query tokens: the last token of the phrase (locate) or of the instruction (policy),
the last token of the prompt, and (locate) the token that writes the first box coordinate, after the reply is started
with BOX. Each map is scored against three masks on the image-token grid: AUROC of the mask's cells against cells off
it, and whether the strongest cell lies on it.

    locate masks: target (asked for), control (another thing in view, not asked for), gripper (not asked for)
    policy masks: object (the task's object), fixture (the fixtures the task refers to), gripper

Also the share of each layer's write that comes from image tokens. Writes attention_<mode>_<size>_<shard>.npz with
auroc and point [examples, masks, queries, layers, heads + 1] (last head index: all heads together), image_share
[examples, queries, layers], x1 (locate: the first coordinate the model writes after BOX and the true one) and the maps
of the first --keep examples.

    python -m qwen_vla.probe_attention --model <checkpoint> --scenes <scenes dir> --output <dir> --mode locate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qwen_vla import scenes, vlm
from qwen_vla.probe_behavior import GROUND

BOX = '```json\n[\n\t{"bbox_2d": ['  # how Qwen starts a grounding reply
MASKS = {'locate': ['target', 'control', 'gripper'], 'policy': ['object', 'fixture', 'gripper']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('locate', 'policy'), default='locate')
    parser.add_argument('--size', type=int, default=768, choices=(256, 768), help='image side (locate)')
    parser.add_argument('--per-image', type=int, default=2, help='things asked per image (locate)')
    parser.add_argument('--keep', type=int, default=40, help='examples whose maps are saved')
    parser.add_argument('--limit', type=int, help='use only the first N scenes')
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    args = parser.parse_args()

    model, processor = vlm.load(args.model)
    device, tokenizer = model.device, processor.tokenizer
    layers = model.model.language_model.layers
    heads = max(model.config.text_config.linear_num_value_heads, model.config.text_config.num_attention_heads)
    size = args.size if args.mode == 'locate' else 256

    examples = []  # images, text, tokens of the content query, and the three masks per image
    for path in sorted(args.scenes.glob('*.npz'))[:args.limit][args.shard[0]::args.shard[1]]:
        scene = scenes.load(path)
        names = [e['name'] for e in scene['entities']]
        gripper = scene['mask'] == (names.index('gripper') if 'gripper' in names else -1)
        frames = [Image.fromarray(f) for f in scene['rgb' if size == 768 else 'rgb_256']]
        if args.mode == 'policy':
            fixture = np.isin(scene['mask'], [names.index(n) for n in scene['fixture_refs'].values() if n in names])
            obj = scene['mask'] == (names.index('obj') if 'obj' in names else -1)
            examples.append({'scene': scene['path'], 'images': frames, 'text': scene['lang'], 'content': scene['lang'],
                             'masks': [obj, fixture, gripper]})
            continue
        for camera in range(3):
            things = scenes.targets(scene, camera)
            order = np.random.default_rng(camera).permutation(len(things))
            for i in order[:args.per_image]:
                t, others = things[i], [o for o in things if o is not things[i] and o['role'] != 'gripper']
                control = others[0]['mask'] if others else np.zeros_like(t['mask'])
                examples.append({
                    'scene': scene['path'], 'camera': scenes.CAMERAS[camera], 'phrase': t['phrase'], 'role': t['role'],
                    'images': [frames[camera]], 'text': GROUND.format(t['phrase']), 'content': f'Locate the {t["phrase"]}',
                    'masks': [t['mask'][None], control[None], (gripper[camera] & ~t['mask'])[None]], 'x1': float(t['box'][0]),
                })

    n, kept = len(examples), {}
    auroc = np.full((n, 3, 3, len(layers), heads + 1), np.nan, dtype=np.float32)
    point = np.full_like(auroc, np.nan)
    image_share = np.full((n, 3, len(layers)), np.nan, dtype=np.float32)
    x1 = np.full((n, 2), np.nan, dtype=np.float32)
    for e, example in enumerate(tqdm(examples)):
        inputs = vlm.chat(processor, [(example['images'], example['text'])]).to(device)
        ids = inputs['input_ids'][0]
        grids = vlm.image_tokens(model, inputs)
        cells = torch.cat([g.flatten() for g in grids])
        # Query tokens: the last token of the content (it follows the last image), the last prompt token, the box token
        content = int(grids[-1][-1, -1]) + 1 + len(tokenizer(example['content'], add_special_tokens=False).input_ids)
        queries = [content, len(ids) - 1]
        if args.mode == 'locate':
            box = tokenizer(BOX, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], box], dim=1)
            for key in ('attention_mask', 'mm_token_type_ids'):
                inputs[key] = torch.cat([inputs[key], torch.full_like(box, key == 'attention_mask')], dim=1)
            queries.append(inputs['input_ids'].shape[1] - 1)
        queries = torch.tensor(queries, device=device)
        traced = vlm.trace(model, inputs, queries)
        rows, q = traced['rows'], len(queries)
        if args.mode == 'locate':
            written = tokenizer.decode(model.lm_head(traced['hidden'][-1, -1]).argmax())
            x1[e] = float(written) if written.strip().isdigit() else np.nan, example['x1']

        # Mask coverage of every image-token cell; a cell is on the mask above 0.2 (or the best cell), off it at 0
        cover = [torch.from_numpy(np.concatenate([scenes.cells(m.astype(np.float32), *g.shape).ravel()
                                                  for m, g in zip(mask, grids)])).to(device) for mask in example['masks']]
        for l, row in enumerate(rows):
            maps = torch.cat([row['head_norm'], row['norm'][:, None]], dim=1)[:, :, cells]  # [Q, heads + 1, cells]
            image_share[e, :q, l] = (row['norm'][:, cells].sum(-1) / row['norm'].sum(-1)).cpu().numpy()
            h = maps.shape[1] - 1
            for m, c in enumerate(cover):
                if c.max() == 0:
                    continue
                on, off = c >= min(0.2, float(c.max())), c == 0
                ranks = maps[:, :, on | off].argsort(-1).argsort(-1).float() + 1
                on_ranks = ranks[:, :, on[on | off]]
                score = (on_ranks.mean(-1) - (on.sum() + 1) / 2) / off.sum()
                hit = (c[maps.argmax(-1)] > 0).float()
                for values, array in ((score, auroc), (hit, point)):
                    array[e, m, :q, l, :h] = values[:, :h].cpu().numpy()
                    array[e, m, :q, l, -1] = values[:, h].cpu().numpy()
            if e < args.keep:
                kept.setdefault(f'maps_{e}', []).append(maps.half().cpu().numpy())
        if e < args.keep:
            kept[f'maps_{e}'] = np.stack([np.pad(m, ((0, 0), (0, heads + 1 - m.shape[1]), (0, 0))) for m in kept[f'maps_{e}']])
            kept[f'cover_{e}'] = torch.stack(cover).cpu().numpy()
            kept[f'grids_{e}'] = np.array([g.shape for g in grids])

    args.output.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in example.items() if k not in ('images', 'masks')} for example in examples]
    np.savez_compressed(
        args.output / f'attention_{args.mode}_{size}_{args.shard[0]}.npz', auroc=auroc, point=point, image_share=image_share, x1=x1,
        masks=MASKS[args.mode], types=[l.block_type for l in layers], meta=json.dumps(meta), **kept,
    )
    best = np.nanmean(auroc[:, 0, 0, :, -1], 0)
    print(f'{n} examples; AUROC of {MASKS[args.mode][0]} from the content token, all heads, by layer:')
    print(' '.join(f'{v:.2f}' for v in best))


if __name__ == '__main__':
    main()

"""Train a VLA (vla.py) on a mixture of robot embodiments (data.py).

    torchrun --nproc_per_node=4 -m qwen_vla.train --mode frozen --vlm <Qwen dir> --index <index dir> --output <run dir>

--mode frozen and adapter keep a frozen bf16 copy of the VLM on every GPU. --mode full trains the VLM as well, sharded
over all GPUs (FSDP) with float32 weights; it stays near the pretrained weights through a small learning rate
(--vlm-lr-scale) and a pull that removes the fraction --pull of its drift every step. The run writes log.jsonl
(training loss, validation flow loss and open-loop action error per embodiment, VLM drift) and checkpoints of what was
trained.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from qwen_vla import vlm
from qwen_vla.data import Mixture
from qwen_vla.vla import VLA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('frozen', 'full', 'adapter'), required=True)
    parser.add_argument('--vlm', required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tap', type=int, required=True, help='VLM layer whose output the action expert reads')
    parser.add_argument('--weights', type=json.loads, required=True, help='sampling weight per embodiment, JSON')
    parser.add_argument('--expert-layers', type=int, default=16)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=16, help='per GPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--vlm-lr-scale', type=float, default=0.05)
    parser.add_argument('--pull', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-batches', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group('nccl')
    args.output.mkdir(parents=True, exist_ok=True)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    def log(record: dict) -> None:
        if rank == 0:
            print(json.dumps(record), flush=True)
            with open(args.output / 'log.jsonl', 'a') as f:
                f.write(json.dumps(record) + '\n')

    # Embodiments: sizes from the index, padded to the largest
    names = list(args.weights)
    sizes = {n: tuple(np.load(args.index / f'{n}.npz')[k].shape[1] for k in ('state_range', 'action_range')) for n in names}
    state_dim, action_dim = (max(s[i] for s in sizes.values()) for i in (0, 1))
    processor = vlm.AutoProcessor.from_pretrained(args.vlm)

    def collate(samples: list[dict]) -> dict:
        batch = {'inputs': vlm.chat(processor, [(s['images'], s['text']) for s in samples]),
                 'state': torch.zeros(len(samples), state_dim), 'action': torch.zeros(len(samples), 16, action_dim),
                 'valid': torch.zeros(len(samples), 16, action_dim), 'embodiment': torch.tensor([names.index(s['embodiment']) for s in samples])}
        for i, s in enumerate(samples):
            batch['state'][i, :len(s['state'])] = torch.from_numpy(s['state'])
            batch['action'][i, :, :s['action'].shape[1]] = torch.from_numpy(s['action'])
            batch['valid'][i, :, :s['action'].shape[1]] = torch.from_numpy(s['valid'])[:, None]
        return batch

    loader = lambda split, seed, workers: iter(DataLoader(
        Mixture(args.index, args.weights, seed=seed, split=split), batch_size=args.batch, num_workers=workers, collate_fn=collate, prefetch_factor=4))
    held_out = [batch for batch, _ in zip(loader('val', 12345 + rank, 2), range(args.eval_batches))]  # the same batches at every evaluation
    batches = loader('train', args.seed * 1000 + rank, args.workers)
    to_device = lambda batch: {k: v.to(device) for k, v in batch.items()}

    # Model. The VLM is cut at the tap before anything is copied.
    backbone, _ = vlm.load(args.vlm, device_map={'': device})
    model = VLA(backbone, args.tap, state_dim, action_dim, len(names), mode=args.mode, num_layers=args.expert_layers).to(device)
    del backbone.lm_head
    if args.mode == 'full':
        # Float32 weights sharded over all GPUs; each block, the rest of the VLM and the expert gather theirs when called
        model.vlm.float()
        for module in list(model.vlm.visual.blocks) + list(model.vlm.language_model.layers) + [model.vlm, model.expert]:
            fully_shard(module)
        step_model = model
    else:
        model.vlm.eval()
        step_model = DistributedDataParallel(model, device_ids=[device.index])
    trained = list(model.expert.parameters()) + list(model.adapter.parameters() if model.adapter else [])
    groups = [{'params': trained, 'lr': args.lr}]
    if args.mode == 'full':
        backbone_params = list(model.vlm.parameters())
        groups.append({'params': backbone_params, 'lr': args.lr * args.vlm_lr_scale})
        anchors = [p.detach().to_local().clone() for p in backbone_params]  # this rank's shard of the pretrained weights
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    schedule = lambda step: min(1, (step + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * step / args.steps)) * 0.9 + 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if rank == 0:
        count = lambda params: sum(p.numel() for p in params) / 1e6
        log({'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'gpus': world, 'embodiments': sizes,
             'million_parameters': {'vlm': count(model.vlm.parameters()), 'expert': count(model.expert.parameters()),
                                    'adapter': count(model.adapter.parameters()) if model.adapter else 0}})

    def evaluate() -> dict:
        """Flow loss and open-loop action error (normalized units) on held-out episodes, per embodiment, over all GPUs."""
        model.eval()
        rng = torch.cuda.get_rng_state(device)
        torch.manual_seed(0)
        totals = torch.zeros(3, len(names), device=device)  # flow loss, action error, samples
        for batch in held_out:
            batch = to_device(batch)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                noise, t = torch.randn_like(batch['action']), torch.rand(len(batch['action']), device=device)
                action = model.sample(batch['inputs'], batch['state'], batch['embodiment'], action_noise=noise)
                context, mask = model.context(batch['inputs'])
                flow = model.expert(batch['action'] + t.view(-1, 1, 1) * (noise - batch['action']), t * 1000, batch['state'], batch['embodiment'], context, mask).float()
            per_sample = lambda x: (x * batch['valid']).sum((1, 2)) / batch['valid'].sum((1, 2))
            for row, values in enumerate((per_sample((flow - (noise - batch['action'])) ** 2), per_sample((action - batch['action']) ** 2), torch.ones_like(t))):
                totals[row].index_add_(0, batch['embodiment'], values)
        dist.all_reduce(totals)
        torch.cuda.set_rng_state(rng, device)
        model.vlm.train(args.mode == 'full')
        model.expert.train()
        return {'val_loss': dict(zip(names, (totals[0] / totals[2]).tolist())), 'val_action_mse': dict(zip(names, (totals[1] / totals[2]).tolist()))}

    def save(step: int) -> None:
        folder = args.output / f'checkpoint-{step}'
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
        full = lambda module: {k: (v.full_tensor() if hasattr(v, 'full_tensor') else v) for k, v in module.state_dict().items()}
        for name, module in (('expert', model.expert), ('adapter', model.adapter), ('vlm', model.vlm if args.mode == 'full' else None)):
            if module is not None:
                state = {k: v.to('cpu', torch.bfloat16 if name == 'vlm' else v.dtype).contiguous() for k, v in full(module).items()}  # collective under FSDP
                rank == 0 and save_file(state, folder / f'{name}.safetensors')
        rank == 0 and (folder / 'config.json').write_text(json.dumps({
            'mode': args.mode, 'vlm': args.vlm, 'tap': args.tap, 'expert_layers': args.expert_layers, 'embodiments': names, 'state_dim': state_dim, 'action_dim': action_dim}))

    model.vlm.train(args.mode == 'full')
    model.expert.train()
    wait = busy = 0.0
    for step in range(args.steps):
        start = time.time()
        batch = to_device(next(batches))
        wait += time.time() - start
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, info = step_model(**batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        info['grad_norm'] = torch.nn.utils.clip_grad_norm_(trained, max_norm=1.0).item()
        if args.mode == 'full':
            info['vlm_grad_norm'] = torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0).item()
        optimizer.step()
        scheduler.step()
        if args.mode == 'full' and args.pull:
            with torch.no_grad():
                for p, anchor in zip(backbone_params, anchors):
                    p.to_local().lerp_(anchor, args.pull)
        busy += time.time() - start

        if (step + 1) % 50 == 0 or step == 0:
            record = {'step': step + 1, **info, 'lr': scheduler.get_last_lr()[0], 'seconds_per_step': busy / (50 if step else 1), 'data_wait': wait / (50 if step else 1)}
            wait = busy = 0.0
            if args.mode == 'full' and (step + 1) % 500 == 0:
                moved = torch.stack([torch.stack([(p.to_local() - a).pow(2).sum(), a.pow(2).sum()]) for p, a in zip(backbone_params, anchors)]).sum(0)
                dist.all_reduce(moved)
                record['vlm_drift'] = (moved[0] / moved[1]).sqrt().item()
            log(record)
        if (step + 1) % args.eval_every == 0 or step == 0:
            log({'step': step + 1, **evaluate()})
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1)
    log({'done': True})
    os._exit(0)  # the data workers never end


if __name__ == '__main__':
    main()

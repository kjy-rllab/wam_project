"""Validation loss of a RoboCasa context-noise checkpoint at fixed context noise levels (tcont_context).

The samples come from the run's own data, in three sets:
  trained  samples the run trained on: the first shard visit of each data worker in a job, whose in-shard shuffle
           starts from the job's data seed (seed + resume step, see BaseTrainer.get_train_dataloader)
  heldout  shards of the same tasks that no worker visited
  unseen   tasks of which no shard was visited
The dataset order and the shard schedules are rebuilt from the run config, so run this on the cluster that trained
(the dataset glob order is the file system's). Every sample is scored at every level with the same noise draws, so
the levels are compared on the same samples.

    torchrun --nproc_per_node=4 eval_utils/robocasa_val_loss.py --model-path <checkpoint dir> --resume-steps 0 9000 18000
"""
import dataclasses
import datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tyro
from hydra.utils import instantiate
from omegaconf import OmegaConf

from groot.vla.model.dreamzero.base_vla import VLA
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform


@dataclasses.dataclass
class Args:
    model_path: str  # training checkpoint, e.g. runs/dreamzero_robocasa_context_noise/checkpoint-25000
    resume_steps: list[int]  # global steps the run's jobs started from, e.g. 0 9000 18000
    train_gpus: int = 64  # data parallel ranks of the run (16 nodes x 4 GPUs)
    # Context noise levels. Powers of 2 stay exact through the bf16 cast of the batch.
    levels: list[float] = dataclasses.field(default_factory=lambda: [0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0])
    tasks: int = 40  # tasks per set at most; trained and heldout share theirs, half atomic and half composite
    samples_per_task: int = 4
    seed: int = 0
    output_dir: str | None = None  # default: <run dir>/val_loss/<checkpoint>


def shard_sets(cfg, global_step: int, resume_steps: list[int], train_gpus: int, tasks: int, seed: int):
    """The run's training mixture, the task of each of its datasets (e.g. atomic/CloseMicrowave), the datasets the run
    sampled, and the shards to evaluate: (set, dataset, shard, data seed of a trained visit, samples trained)."""
    # Shard visits up to this checkpoint. Schedule entry k * slots + j is the k-th shard of data worker j, which takes
    # per_visit samples from it and gives one batch every `workers` steps.
    mixture = instantiate(cfg.train_dataset)
    task = [str(Path(*Path(ds.dataset_path).parts[-4:-2])) for ds in mixture.datasets]
    per_visit = int(mixture.datasets[0].num_steps_per_shard * mixture.shard_sampling_rate)
    workers, batch_size = cfg.training_args.dataloader_num_workers, cfg.training_args.per_device_train_batch_size
    slots, visit_steps = train_gpus * workers, per_visit * workers // batch_size
    base_seed = mixture.seed
    visited, first_visits = set(), {}  # first_visits: dataset -> [(shard, data seed, samples trained)]
    for start, end in zip(resume_steps, resume_steps[1:] + [global_step]):
        end = min(end, global_step)
        if end <= start:
            continue
        mixture.reset_seed(base_seed + start)
        schedule = [tuple(entry) for entry in mixture.shards_sample_schedule[: math.ceil((end - start) / visit_steps) * slots]]
        visited.update(schedule)
        for d, s in schedule[:slots]:
            first_visits.setdefault(d, []).append((s, base_seed + start, min(per_visit, (end - start) // workers * batch_size)))

    rng = np.random.default_rng(seed)
    unvisited = [[s for s in range(ds.num_shards) if (d, s) not in visited] for d, ds in enumerate(mixture.datasets)]
    units = []
    for kind in ("atomic", "composite"):
        both = [d for d in sorted(first_visits) if unvisited[d] and task[d].startswith(kind)]
        for d in rng.permutation(both)[: tasks // 2]:
            units.append(("trained", d, *first_visits[d][rng.integers(len(first_visits[d]))]))
            units.append(("heldout", d, rng.choice(unvisited[d]), None, None))
    seen = {d for d, _ in visited}
    for d in rng.permutation([d for d in range(len(task)) if d not in seen])[: tasks]:
        units.append(("unseen", d, rng.integers(mixture.datasets[d].num_shards), None, None))
    return mixture, task, seen, units


def main(args: Args):
    os.environ.setdefault("ATTENTION_BACKEND", "torch")  # as in training
    dist.init_process_group("gloo", timeout=datetime.timedelta(hours=12))  # only for the final barrier
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    model_dir = Path(args.model_path)
    output_dir = Path(args.output_dir or model_dir.parent / "val_loss" / model_dir.name)
    cfg = OmegaConf.load(model_dir / "experiment_cfg" / "conf.yaml")
    global_step = json.loads((model_dir / "trainer_state.json").read_text())["global_step"]
    mixture, task, seen, units = shard_sets(cfg, global_step, args.resume_steps, args.train_gpus, args.tasks, args.seed)
    if rank == 0:
        print(f"Checkpoint step {global_step}: {len(seen)} tasks sampled, {len(task) - len(seen)} never; "
              f"evaluating {len(units)} shards x {args.samples_per_task} samples x {len(args.levels)} levels", flush=True)

    # As GrootSimPolicy loads it, one rank at a time: a loading rank holds about 120 GB of CPU memory, and on GH200 the
    # CPU memory spills into GPU memory once full (four ranks at once ran a GPU out of memory)
    for r in range(world):
        if r == rank:
            model = VLA.from_pretrained(args.model_path)
            model.eval()
            model.requires_grad_(False)
            if cfg.get("eval_bf16", False):
                model = model.to(dtype=torch.bfloat16)
            model.to("cuda")
            model.post_initialize()
        dist.barrier()
    collator = instantiate(cfg.data_collator)
    shift = model.action_head.scheduler.shift

    records = []
    for u in range(rank, len(units), world):
        name, d, shard, data_seed, trained = units[u]
        dataset = mixture.datasets[d]
        # Video augmentation off; DreamTransform stays in training mode, which adds the action targets
        for t in dataset.transforms.transforms:
            if not isinstance(t, DreamTransform):
                t.eval()
        dataset.start_cache_shard(shard)
        dataset.finish_cache_shard()
        steps = []  # as in ShardedLeRobotMixtureDataset.__iter__
        for trajectory_id in dataset.get_trajectories_in_shard():
            length = dataset.trajectory_lengths[dataset.get_trajectory_index(trajectory_id)]
            allowed = dataset.step_filter[trajectory_id]
            allowed = allowed[allowed <= (length if mixture.allow_padding_at_end else length - dataset.max_delta_index)]
            steps += [(trajectory_id, i) for i in allowed]
        if data_seed is not None:  # the worker's shuffle of this shard; it trained on the first samples
            np.random.default_rng(data_seed).shuffle(steps)
            steps = steps[:trained]
        for j, k in enumerate(np.random.default_rng([args.seed, u]).permutation(len(steps))[: args.samples_per_task]):
            trajectory_id, i = steps[k]
            data = dataset.get_step_data(trajectory_id, {key: delta + i for key, delta in dataset.delta_indices.items()})
            if data is None:
                continue
            batch = collator([dataset.transforms(data)])
            for level in args.levels:
                torch.manual_seed(args.seed * 1_000_000 + u * 1_000 + j)  # the same noise draws at every level
                with torch.no_grad():
                    out = model(batch | {"tcont_context": torch.tensor([level])})
                records.append({
                    "set": name, "task": task[d], "dataset": int(d), "shard": int(shard), "trajectory": int(trajectory_id),
                    "step": int(i), "tcont": level, "sigma": shift * level / (1 + (shift - 1) * level),
                    "dynamics_loss": out["dynamics_loss"].item(), "action_loss": out["action_loss"].item(),
                })
        dataset.delete_cached_shard()
        print(f"Rank {rank}: shard {u + 1}/{len(units)} done ({name}, {task[d]}, shard {shard})", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"records_{rank}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    dist.barrier()
    if rank > 0:
        return

    records = []
    for r in range(world):
        part = output_dir / f"records_{r}.jsonl"
        records += [json.loads(line) for line in part.read_text().splitlines()]
        part.unlink()
    (output_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    # Mean and standard error per set (and per task kind) and level
    summary = {}
    for group in ("trained", "heldout", "unseen") + tuple(f"{s}/{k}" for s in ("trained", "heldout", "unseen") for k in ("atomic", "composite")):
        rows = [r for r in records if group in (r["set"], f"{r['set']}/{r['task'].split('/')[0]}")]
        if not rows:
            continue
        summary[group] = {}
        for level in args.levels:
            at = [r for r in rows if r["tcont"] == level]
            stats = {"sigma": at[0]["sigma"], "n": len(at)}
            for key in ("dynamics_loss", "action_loss"):
                x = np.array([r[key] for r in at])
                stats |= {key: x.mean(), f"{key}_se": x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")}
            summary[group][str(level)] = stats
    (output_dir / "summary.json").write_text(json.dumps({"checkpoint": str(model_dir), "global_step": global_step,
                                                         "summary": summary}, indent=2))
    for key in ("dynamics_loss", "action_loss"):
        print(f"\n{key} (tcont / sigma)")
        print(f"{'set':<20}{'n':>5}" + "".join(f"{f'{l:g}/{shift * l / (1 + (shift - 1) * l):.2f}':>14}" for l in args.levels))
        for group, by_level in summary.items():
            n = by_level[str(args.levels[0])]["n"]
            print(f"{group:<20}{n:>5}" + "".join(f"{by_level[str(l)][key]:>14.4f}" for l in args.levels))
    print(f"\nSaved {output_dir}/records.jsonl and summary.json", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))

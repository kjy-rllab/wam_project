"""Video and action samples of a RoboCasa context-noise checkpoint at fixed context noise levels (tcont_context), to
see whether noising the context interpolates between the prediction given the context (level 0) and the marginal
(level 1: the observed frames are pure noise, only the instruction and the robot state stay), and whether the videos in
between are still plausible videos.

Contexts come from the heldout set of robocasa_val_loss.py (shards of trained tasks that no data worker visited): from
each heldout shard, trajectories with a random step i whose instruction holds up to step i+48. A context is played as
the simulation server plays an episode: a first call with the frame of step i, then a call with the frames of steps
i..i+24 (stride 3), each predicting 24 actions and the 8 frames that go with them, so the prediction covers steps
i+3..i+48 (frames 1..16). Seed k fixes the initial noise of the video and the actions and the context noise, so the
levels of a seed see the same draws. Each call's video latents are decoded after the GT latents it was conditioned on.

The first-frame condition y of Wan I2V is the VAE encoding of the first frame followed by zero frames. The context noise
replaces its first latent (as in training), but the causal VAE encoder carries the first frame into the latents of the
zero frames. With --y-tail swap those latents come from the first frame of another context, half the context list away
(another task), to see whether the prediction then shows the other scene; the relative change of each latent of y under
the swap is saved as tail_change.

Saved per context: samples/<context>.npz with the predicted and GT actions, CLIP image embeddings (the model's image
encoder) of the predicted, GT and VAE-reconstructed frames of each view, and their pixel statistics (error to GT,
sharpness, motion); latents/<context>.pt with the predicted video latents; videos/<context>_seed<k>.mp4 for seeds 0
and 1, rows GT and then the levels, the views as the model sees them.

    torchrun --nproc_per_node=4 eval_utils/robocasa_context_noise_samples.py --model-path <checkpoint dir> --resume-steps 0 9000 18000
"""
import dataclasses
import datetime
import json
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tyro
from omegaconf import OmegaConf
from tianshou.data import Batch

from eval_utils.robocasa_val_loss import shard_sets
from groot.vla.data.dataset.lerobot import LeRobotSingleDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy, unsqueeze_dict_values


@dataclasses.dataclass
class Args:
    model_path: str  # training checkpoint, e.g. runs/dreamzero_robocasa_context_noise/checkpoint-25000
    resume_steps: list[int]  # global steps the run's jobs started from, e.g. 0 9000 18000
    train_gpus: int = 64  # data parallel ranks of the run
    # Context noise levels; sigma = 5t / (1 + 4t), so 1/8 is SNR 2 and 1/4 is SNR 0.36
    levels: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2, 1.0])
    seeds: int = 5  # samples per context and level
    tasks: int = 24  # heldout tasks, half atomic and half composite
    contexts_per_task: int = 2  # trajectories of the task's heldout shard
    seed: int = 0
    # Latents of y after the first (the encoded zero frames): "own" as in training, or "swap" for another context's
    y_tail: Literal["own", "swap"] = "own"
    output_dir: str | None = None  # default: <run dir>/context_noise_samples/<checkpoint>[_y_tail_swap]


def main(args: Args):
    os.environ.setdefault("ATTENTION_BACKEND", "torch")  # as in training
    torch._dynamo.config.recompile_limit = 800  # as serve_robocasa.py
    torch.set_grad_enabled(False)
    dist.init_process_group("gloo", timeout=datetime.timedelta(hours=12))  # only for barriers
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    model_dir = Path(args.model_path)
    output_dir = Path(args.output_dir or model_dir.parent / "context_noise_samples" /
                      (model_dir.name + ("" if args.y_tail == "own" else f"_y_tail_{args.y_tail}")))
    cfg = OmegaConf.load(model_dir / "experiment_cfg" / "conf.yaml")
    global_step = json.loads((model_dir / "trainer_state.json").read_text())["global_step"]
    mixture, task, _, units = shard_sets(cfg, global_step, args.resume_steps, args.train_gpus, args.tasks, args.seed)

    # Contexts: (dataset, trajectory, step, instruction). The training dataset class resamples the steps it is given
    # around the first one, so the data is read with the plain getters of LeRobotSingleDataset.
    horizon = len(cfg.train_dataset.all_modality_configs.robocasa_panda_omron.action.delta_indices)
    contexts = []
    for u, (kind, d, shard, _, _) in enumerate(units):
        if kind != "heldout":
            continue
        dataset, rng, found = mixture.datasets[d], np.random.default_rng([args.seed, u]), 0
        language_key = dataset.modality_keys["language"][0]
        for trajectory in rng.permutation(dataset.sharded_trajectories[shard]):
            if found == args.contexts_per_task:
                break
            length = dataset.trajectory_lengths[dataset.get_trajectory_index(trajectory)]
            if length <= 2 * horizon:
                continue
            dataset.curr_traj_data = LeRobotSingleDataset.get_trajectory_data(dataset, trajectory)
            i = int(rng.integers(length - 2 * horizon))
            language = set(dataset.get_language(trajectory, language_key, np.arange(i, i + 2 * horizon + 1)))
            if len(language) == 1:
                contexts.append((d, int(trajectory), i, language.pop()))
                found += 1
    if rank == 0:
        print(f"{len(contexts)} contexts x {len(args.levels)} levels x {args.seeds} seeds, y tail {args.y_tail}", flush=True)

    # As GrootSimPolicy loads it, one rank at a time (see robocasa_val_loss.py)
    for r in range(world):
        if r == rank:
            policy = GrootSimPolicy(EmbodimentTag("robocasa_panda_omron"), args.model_path, device="cuda")
        dist.barrier()
    head, keys = policy.trained_model.action_head, policy.modality_configs
    visual = head.image_encoder.model.visual
    block = head.num_frame_per_block
    stride = horizon // (4 * block)  # Wan VAE: 4 frames per latent
    steps = np.arange(0, 2 * horizon + 1, stride)  # frames of the two calls, from step i
    observed = len(steps) // 2 + 1  # frames the second call observes
    laplacian = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda")[None, None]
    encode_image = head.encode_image  # clip feature, y and first latent of a first frame
    if args.y_tail == "swap":  # the model builds y at the call that starts the KV cache; partner_y is set per context
        def encode_image_swapped(image, num_frames, height, width):
            clip_feas, ys, new_image = encode_image(image, num_frames, height, width)
            ys[:, 4:, 1:] = partner_y[:, :, 1:]
            return clip_feas, ys, new_image
        head.encode_image = encode_image_swapped

    def views(video):
        """Frames (T, view, C, 224, 224) in 0..255 of the model's video (C, T, 448, 448) in [-1, 1], whose 2x2 grid is
        [left, wrist; right, black]: left, right, wrist."""
        x = ((video.float().transpose(0, 1) + 1) * 127.5).round().clamp(0, 255)
        h, w = x.shape[-2] // 2, x.shape[-1] // 2
        return torch.stack([x[..., :h, :w], x[..., h:, :w], x[..., :h, w:]], 1)

    def embed(frames):
        """CLIP image embeddings (T, view, 1024) of frames (T, view, C, H, W), with every layer of the model's encoder."""
        x = head.image_encoder.encode_image([frames.flatten(0, 1).to(torch.bfloat16) / 127.5 - 1])  # all but the last
        x = visual.post_norm(visual.transformer[-1](x)[:, 0]) @ visual.head
        return F.normalize(x.float(), dim=-1).unflatten(0, frames.shape[:2]).half().cpu().numpy()

    def stats(frames, previous, gt):
        """Per frame and view: mean absolute error to GT at full and 1/8 resolution, sharpness (mean |Laplacian| of the
        gray image), and motion from the previous frame (mean absolute difference at 1/8 resolution, mean optical flow
        magnitude)."""
        low = lambda x: F.avg_pool2d(x.flatten(0, 1), 8).abs().mean((1, 2, 3)).view(x.shape[:2])
        gray = frames.mean(2, keepdim=True)
        a, b = previous.mean(2).byte().cpu().numpy(), gray[:, :, 0].byte().cpu().numpy()
        flow = [[np.linalg.norm(cv2.calcOpticalFlowFarneback(a[t, v], b[t, v], None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=-1).mean()
                 for v in range(a.shape[1])] for t in range(a.shape[0])]
        return {"mae": (frames - gt).abs().mean((2, 3, 4)).cpu().numpy(), "mae_low": low(frames - gt).cpu().numpy(),
                "sharpness": F.conv2d(gray.flatten(0, 1), laplacian).abs().mean((1, 2, 3)).view(frames.shape[:2]).cpu().numpy(),
                "motion_low": low(frames - previous).cpu().numpy(), "flow": np.array(flow, dtype=np.float32)}

    for c in range(rank, len(contexts), world):
        d, trajectory, i, language = contexts[c]
        name = f"{task[d].replace('/', '_')}_traj{trajectory}_step{i}"
        if (output_dir / "samples" / f"{name}.npz").exists():
            continue
        start = time.time()
        dataset = mixture.datasets[d]
        dataset.curr_traj_data = LeRobotSingleDataset.get_trajectory_data(dataset, trajectory)
        video = {k: LeRobotSingleDataset.get_video(dataset, trajectory, k, i + steps) for k in keys.video.modality_keys}
        state = {k: LeRobotSingleDataset.get_state_or_action(dataset, trajectory, "state", k, np.array([i, i + horizon]))
                 for k in keys.state.modality_keys}
        gt_actions = np.concatenate([LeRobotSingleDataset.get_state_or_action(
            dataset, trajectory, "action", k, np.arange(i, i + 2 * horizon)) for k in keys.action.modality_keys], axis=1)
        calls = [{k: v[:1] for k, v in video.items()} | {k: v[:1] for k, v in state.items()},
                 {k: v[:observed] for k, v in video.items()} | {k: v[1:] for k, v in state.items()}]
        language_key = keys.language.modality_keys[0]
        calls = [obs | {language_key: language} for obs in calls]

        # GT frames as the model sees them (eval transform: center crop, resize, grid) and their VAE latents
        images = torch.as_tensor(policy.eval_transform(unsqueeze_dict_values(calls[0] | video))["images"])
        gt_video = (images.cuda().permute(0, 4, 1, 2, 3).float() / 127.5 - 1).to(torch.bfloat16)
        gt_latents = head.vae.encode(gt_video)
        gt, recon = views(gt_video[0]), views(head.vae.decode(gt_latents)[0])
        out = {"clip_gt": embed(gt), "clip_recon": embed(recon), "actions_gt": gt_actions,
               "mae_copy": (gt[[0] * (observed - 1) + [observed - 1] * (observed - 1)] - gt[1:]).abs().mean((2, 3, 4)).cpu().numpy()}
        out |= {f"{k}_gt": v for k, v in stats(gt[1:], gt[:-1], gt[1:]).items() if k in ("sharpness", "motion_low", "flow")}
        out |= {f"{k}_recon": v for k, v in stats(recon[1:], recon[:-1], gt[1:]).items()}
        if args.y_tail == "swap":  # y of the partner's first frame as the model sees it
            dp, tp, ip, _ = contexts[(c + len(contexts) // 2) % len(contexts)]
            partner = mixture.datasets[dp]
            partner.curr_traj_data = LeRobotSingleDataset.get_trajectory_data(partner, tp)
            frame = {k: LeRobotSingleDataset.get_video(partner, tp, k, np.array([ip])) for k in keys.video.modality_keys}
            frame = torch.as_tensor(policy.eval_transform(unsqueeze_dict_values(calls[0] | frame))["images"])
            frame = (frame.cuda().permute(0, 4, 1, 2, 3).float() / 127.5 - 1).to(torch.bfloat16)
            own_y, partner_y = (encode_image(x.transpose(1, 2), head.num_frames, *x.shape[-2:])[1][:, 4:]
                                for x in (gt_video[:, :, :1], frame))
            norm = lambda y: torch.linalg.vector_norm(y.float(), dim=(1, 3, 4))[0]
            out |= {"partner": f"{task[dp].replace('/', '_')}_traj{tp}_step{ip}",
                    "tail_change": (norm(own_y - partner_y) / norm(own_y)).cpu().numpy()}

        samples, latents, videos = [], [], {}
        for seed in range(args.seeds):
            for level in args.levels:
                head.seed = seed  # initial noise of the video and the actions; the context noise uses seed + 1
                results = [policy.lazy_joint_forward_causal(Batch(obs=dict(obs)), tcont_context=level) for obs in calls]
                first, second = (video_pred[:, :, -block:] for _, video_pred in results)
                pred = views(torch.cat([head.vae.decode(torch.cat([gt_latents[:, :, :1], first], 2))[0, :, 1:],
                                        head.vae.decode(torch.cat([gt_latents[:, :, :1 + block], second], 2))[0, :, 1 + 4 * block:]], 1))
                previous = torch.cat([gt[:1], pred[:observed - 2], gt[observed - 1:observed], pred[observed - 1:-1]])
                actions = np.concatenate([np.concatenate([np.asarray(result.act[k], dtype=np.float32).reshape(horizon, -1)
                                                          for k in keys.action.modality_keys], axis=1) for result, _ in results])
                samples.append(stats(pred, previous, gt[1:]) | {"clip": embed(pred), "actions": actions})
                latents.append(torch.cat([first, second], 2)[0].cpu())
                if seed < 2:
                    videos.setdefault(seed, []).append(pred.byte())
        shape = (args.seeds, len(args.levels))
        out |= {k: np.stack([s[k] for s in samples]).reshape(shape + samples[0][k].shape) for k in samples[0]}
        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "latents").mkdir(parents=True, exist_ok=True)
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)
        torch.save({"pred": torch.stack(latents).unflatten(0, shape), "gt": gt_latents[0].cpu()}, output_dir / "latents" / f"{name}.pt")

        # Rows GT and the levels, columns left, right and wrist; frame 0 is the observation in every row
        for seed, preds in videos.items():
            rows = [gt.byte()] + [torch.cat([gt[:1].byte(), p]) for p in preds]  # each T, view, C, H, W
            grid = torch.stack(rows, 1).permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)  # T, C, rows * H, views * W
            grid = np.ascontiguousarray(grid.permute(0, 2, 3, 1).cpu().numpy())
            for r, label in enumerate(["GT"] + [f"tcont {Fraction(level).limit_denominator(64)}" for level in args.levels]):
                for frame in grid:
                    cv2.putText(frame, label, (4, r * gt.shape[-2] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            iio.imwrite(output_dir / "videos" / f"{name}_seed{seed}.mp4", grid, plugin="pyav", fps=20 / stride, codec="libx264")
        np.savez(output_dir / "samples" / f"{name}.npz", **out, task=task[d], trajectory=trajectory, step=i, language=language,
                 y_tail=args.y_tail, levels=np.array(args.levels), steps=steps[1:], views=np.array([k.split(".", 1)[1] for k in keys.video.modality_keys]))
        print(f"Rank {rank}: context {c + 1}/{len(contexts)} {name} done in {time.time() - start:.0f} s", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main(tyro.cli(Args))

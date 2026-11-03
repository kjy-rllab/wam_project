"""Serve DreamZero to the RoboCasa (PandaOmron) client over the RLDX ZeroMQ protocol (ping, get_modality_config,
reset, get_action; msgpack with .npy arrays).

The client (robocasa_workspace/server_inference, tcp://) sends the frames picked by video.delta_indices from its
per-step history. As in training, an action chunk of 24 steps pairs with its 8 frames at stride 3, so every call
after the first gets those 8 frames plus the frame before them, and the client must run whole chunks
(--replan-steps 24). The first call of an episode uses only the current frame, which starts a new KV cache.

get_video returns the video the model predicted along with the actions of the last session: its video latents decoded
by the VAE, uint8 frames of the views' 2x2 grid [left, wrist; right, black]. Frame 0 is the first observation, then
each call adds the 8 frames of its action chunk, so frame i is the prediction for step 3i.

    torchrun --nproc_per_node=1 eval_utils/serve_robocasa.py --model-path <checkpoint dir> [--tcont-context 0.3]

With --nproc_per_node=2 the two CFG branches run on separate GPUs.
"""
import dataclasses
import datetime
import io
import os
import traceback

import msgpack
import numpy as np
import torch
import torch.distributed as dist
import tyro
import zmq
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


@dataclasses.dataclass
class Args:
    model_path: str  # training checkpoint, e.g. runs/dreamzero_robocasa_context_noise/checkpoint-7000
    port: int = 5555
    # Context noise level in [0, 1] of the observations written to the KV cache; None is the original clean inference.
    # A request can set its own level with options["tcont_context"].
    tcont_context: float | None = None


def encode(x):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(x), allow_pickle=False)
    return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}


def decode(d):
    return np.load(io.BytesIO(d["as_npy"]), allow_pickle=False) if "__ndarray_class__" in d else d


def main(args: Args):
    os.environ.setdefault("ATTENTION_BACKEND", "torch")  # as in training
    torch._dynamo.config.recompile_limit = 800  # as the DROID servers; causal chunks take several shapes
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("ip",))
    policy = GrootSimPolicy(EmbodimentTag("robocasa_panda_omron"), args.model_path, device="cuda", device_mesh=mesh)
    group = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=30))  # ranks wait here between requests

    if dist.get_rank() > 0:
        while True:
            message = [None]
            dist.broadcast_object_list(message, src=0, group=group)
            obs, tcont = message[0]
            policy.lazy_joint_forward_causal(Batch(obs=obs), tcont_context=tcont)

    cfg = policy.modality_configs
    horizon = len(cfg.action.delta_indices)
    block = policy.trained_model.action_head.num_frame_per_block  # latent frames predicted per call
    stride = horizon // (4 * block)  # Wan VAE: 4 frames per latent
    names = lambda keys: [k.split(".", 1)[1] for k in keys]
    modality = {
        "video": {"modality_keys": names(cfg.video.modality_keys), "delta_indices": list(range(-horizon, 1, stride))},
        "state": {"modality_keys": names(cfg.state.modality_keys), "delta_indices": [0]},
        "action": {"modality_keys": names(cfg.action.modality_keys), "delta_indices": list(range(horizon))},
        "language": {"modality_keys": list(cfg.language.modality_keys), "delta_indices": [0]},
    }
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Serving {args.model_path} on port {args.port} (tcont_context={args.tcont_context}); "
          f"run the client with --replan-steps {horizon}", flush=True)

    session, latents = None, []
    while True:
        request = msgpack.unpackb(socket.recv(), object_hook=decode, raw=False)
        try:
            endpoint, data = request["endpoint"], request.get("data")
            if endpoint in ("ping", "reset"):
                response = {"status": "ok"}
            elif endpoint == "get_modality_config":
                response = modality
            elif endpoint == "get_action":
                options, observation = data["options"], data["observation"]
                first = options["reset_memory"][0]
                if first:
                    session = options["session_ids"][0]
                elif options["session_ids"][0] != session:
                    raise RuntimeError("another client session took over the policy; reset this episode")
                # Unbatch. Before the first step the history is only padding with the current frame.
                obs = {k: v[0, -1:] if first and k.startswith("video.") else v[0]
                       for k, v in observation.items() if k.startswith(("video.", "state."))}
                obs |= {k: observation[k][0] for k in cfg.language.modality_keys}
                tcont = options.get("tcont_context", args.tcont_context)
                dist.broadcast_object_list([(obs, tcont)], src=0, group=group)
                result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs), tcont_context=tcont)
                # The session's predicted video latents, for get_video. A call that starts a KV cache also returns the
                # latent of its first frame, which only the first call of the episode keeps.
                latents = [video_pred] if first else latents + [video_pred[:, :, -block:]]
                actions = {k: np.asarray(v, dtype=np.float32).reshape(horizon, -1) for k, v in result.act.items()}
                # The data stores these as -1/1; the client takes the Gym convention (0/1, closed at >= 0.5)
                for k in ("action.gripper_close", "action.control_mode"):
                    actions[k] = (actions[k] + 1) / 2
                response = [actions, {}]
            elif endpoint == "get_video":
                with torch.inference_mode():
                    video = policy.trained_model.action_head.vae.decode(torch.cat(latents, dim=2))[0]  # C T H W
                response = ((video.float() + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            else:
                raise ValueError(f"unknown endpoint {endpoint!r}")
        except Exception as e:
            traceback.print_exc()
            response = {"error": f"{type(e).__name__}: {e}"}
        socket.send(msgpack.packb(response, default=encode, use_bin_type=True))


if __name__ == "__main__":
    main(tyro.cli(Args))

"""Serve a trained VLA to the RoboCasa (PandaOmron) client over the RLDX ZeroMQ protocol, as
dreamzero/eval_utils/serve_robocasa.py does for DreamZero (ping, get_modality_config, reset, get_action).

Every request carries the current frame of the three cameras, the state and the instruction; the reply is a chunk of
16 actions. Run the client with --replan-steps 16 or fewer (dreamzero/eval_utils/robocasa_sim_eval.py).

    python -m qwen_vla.serve_robocasa --checkpoint <run dir>/checkpoint-20000 [--port 5555]
"""
import argparse
import io
import json
import traceback
from pathlib import Path

import msgpack
import numpy as np
import torch
import zmq
from PIL import Image
from safetensors.torch import load_file

from qwen_vla import vlm
from qwen_vla.vla import VLA

CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']  # as data.EMBODIMENTS['robocasa']
STATE = [('base_position', 3), ('base_rotation', 4), ('end_effector_position_relative', 3),
         ('end_effector_rotation_relative', 4), ('gripper_qpos', 2)]  # order of observation.state in the data
ACTION = [('base_motion', 4), ('control_mode', 1), ('end_effector_position', 3), ('end_effector_rotation', 3), ('gripper_close', 1)]


def encode(x):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(x), allow_pickle=False)
    return {'__ndarray_class__': True, 'as_npy': buffer.getvalue()}


def decode(d):
    return np.load(io.BytesIO(d['as_npy']), allow_pickle=False) if '__ndarray_class__' in d else d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    config = json.loads((args.checkpoint / 'config.json').read_text())
    backbone, processor = vlm.load(config['vlm'])
    model = VLA(backbone, config['tap'], config['state_dim'], config['action_dim'], len(config['embodiments']),
                mode=config['mode'], num_layers=config['expert_layers'])
    for name, module in (('expert', model.expert), ('adapter', model.adapter), ('vlm', model.vlm)):
        if (args.checkpoint / f'{name}.safetensors').exists():
            module.load_state_dict(load_file(args.checkpoint / f'{name}.safetensors'))
    model.cuda().eval()
    embodiment = torch.tensor([config['embodiments'].index('robocasa')], device='cuda')
    state_range, action_range = (np.array(config['ranges']['robocasa'][k], dtype=np.float32) for k in ('state_range', 'action_range'))

    modality = {
        'video': {'modality_keys': CAMERAS, 'delta_indices': [0]},
        'state': {'modality_keys': [name for name, _ in STATE], 'delta_indices': [0]},
        'action': {'modality_keys': [name for name, _ in ACTION], 'delta_indices': list(range(16))},
        'language': {'modality_keys': ['annotation.human.task_description'], 'delta_indices': [0]},
    }
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f'tcp://*:{args.port}')
    print(f'Serving {args.checkpoint} on port {args.port}; run the client with --replan-steps 16 or fewer', flush=True)
    while True:
        request = msgpack.unpackb(socket.recv(), object_hook=decode, raw=False)
        try:
            endpoint, data = request['endpoint'], request.get('data')
            if endpoint in ('ping', 'reset'):
                response = {'status': 'ok'}
            elif endpoint == 'get_modality_config':
                response = modality
            elif endpoint == 'get_action':
                observation = data['observation']
                images = [Image.fromarray(observation[f'video.{camera}'][0, -1]).resize((config['image_size'],) * 2, Image.BILINEAR) for camera in CAMERAS]
                state = np.concatenate([observation[f'state.{name}'][0, -1] for name, _ in STATE]).astype(np.float32)
                state = np.clip(2 * (state - state_range[0]) / np.maximum(state_range[1] - state_range[0], 1e-6) - 1, -5, 5)
                padded = torch.zeros(1, config['state_dim'], device='cuda')
                padded[0, :len(state)] = torch.from_numpy(state)
                inputs = vlm.chat(processor, [(images, observation['annotation.human.task_description'][0])]).to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    action = model.sample(inputs, padded, embodiment)[0, :, :action_range.shape[1]].float().cpu().numpy()
                action = (action.clip(-1, 1) + 1) / 2 * (action_range[1] - action_range[0]) + action_range[0]
                actions, start = {}, 0
                for name, size in ACTION:
                    actions[f'action.{name}'] = action[:, start:start + size].astype(np.float32)
                    start += size
                # The data stores these as -1/1; the client takes the Gym convention (0/1, closed at >= 0.5)
                for name in ('action.gripper_close', 'action.control_mode'):
                    actions[name] = (actions[name] + 1) / 2
                response = [actions, {}]
            else:
                raise ValueError(f'unknown endpoint {endpoint!r}')
        except Exception as e:
            traceback.print_exc()
            response = {'error': f'{type(e).__name__}: {e}'}
        socket.send(msgpack.packb(response, default=encode, use_bin_type=True))


if __name__ == '__main__':
    main()

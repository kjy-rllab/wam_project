"""Robot demonstrations in LeRobot format (v2.1 and v3.0) as one stream of (images, instruction, state, action chunk).

An embodiment is a group of datasets with the same cameras, state and action. `python -m qwen_vla.data <data root>
<index dir> [embodiment ...]` reads their tables once into <index dir>/<embodiment>/: every frame's state and action,
the episodes (first frame, instruction, video file and start time per camera) and the 1st and 99th percentile of each
state and action dimension, which scale them to [-1, 1]. `Mixture` then samples frames across embodiments and decodes
only the images it needs.

A sample at frame t has the cameras' images at t, the state at t and the actions at t, t + stride, ..., (HORIZON
steps; stride brings every embodiment to 10-20 Hz). Past the end of the episode the last action repeats and `valid`
is False.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

HORIZON = 16
HOLDOUT = 50
# glob under the data root, camera keys, state keys, action keys, frames between actions of a chunk
EMBODIMENTS = {
    'robocasa': ('robocasa/v1.0/pretrain/*/*/*/lerobot', ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
                 ['observation.state'], ['action'], 1),  # PandaOmron, 20 Hz, delta end effector + base
    'aloha': ('real/lerobot__aloha_*', ['cam_high', 'cam_left_wrist', 'cam_right_wrist'], ['observation.state'], ['action'], 3),  # 50 Hz, joints
    'agilex': ('real/BAAI-DataCube__*agilex*', ['camera_front', 'camera_left_wrist', 'camera_right_wrist'],
               ['observation.states.joint_position_left', 'observation.states.joint_position_right'],
               ['actions.joint_position_left', 'actions.joint_position_right'], 2),  # Cobot Magic, 30 Hz, joints
    'fr3_dual': ('real/BAAI-DataCube__*fr3_dual*', ['camera_front', 'camera_left', 'camera_right'],
                 ['observation.states.joint_position'], ['actions.joint_position'], 2),  # two Franka FR3, 30 Hz, joints
    # Tien Kung humanoid, 30 Hz, both arms' joints: with one closure per hand (gello teleoperation: names end in
    # gello_1rgb or tkg_1rgb) or with six finger joints per hand (xsens motion capture: xsens_1rgb or tkx_1rgb)
    'tienkung_gello': ('real/BAAI-DataCube__*[go]_1rgb_*', ['camera_top'], ['observation.states.joint_position'], ['actions.joint_position'], 2),
    'tienkung_xsens': ('real/BAAI-DataCube__*[sx]_1rgb_*', ['camera_top'], ['observation.states.joint_position', 'observation.states.end_effector'],
                       ['actions.joint_position', 'actions.end_effector'], 2),
    'ur5_berkeley': ('real/lerobot__berkeley_autolab_ur5', ['image', 'hand_image'], ['observation.state'], ['action'], 1),  # 5 Hz, delta end effector
    'ur5_table30': ('real/Traly__RoboChallenge-lerobot/*', ['global_image', 'wrist_image'], ['observation.state'], ['action'], 3),  # 30 Hz, end-effector pose
    # RoboMIND UR5e: the recorded follower state is frozen in most episodes, so the state is left out
    'ur5_robomind': ('real/Traly__RoboMIND-lerobot/ur_1rgb', ['camera_top'], [], ['actions.joint_position'], 2),  # 30 Hz, leader joints
}


def index_dataset(job: tuple) -> dict | None:
    """State, action and episode table of one LeRobot dataset."""
    root, cameras, state_keys, action_keys = job
    info = json.loads((root / 'meta/info.json').read_text())
    keys = {c: next((k for k in info['features'] if k.endswith('.' + c) and info['features'][k]['dtype'] == 'video'), None) for c in cameras}
    if None in keys.values() or not any((root / 'meta').glob('episodes*')):  # a few uploads have no episode table
        return None
    stack = lambda table, names: np.concatenate([np.stack(table[n].to_numpy(zero_copy_only=False)).reshape(len(table), -1) for n in names], 1).astype(np.float32) \
        if names else np.zeros((len(table), 0), np.float32)
    columns = state_keys + action_keys + ['episode_index', 'task_index']
    if info['codebase_version'].startswith('v2'):
        episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines()]
        tables = [pq.read_table(root / info['data_path'].format(episode_chunk=e['episode_index'] // info['chunks_size'], episode_index=e['episode_index']), columns=columns) for e in episodes]
        lengths, tasks = [len(t) for t in tables], [e['tasks'][0] for e in episodes]
        files = [[str(root / info['video_path'].format(episode_chunk=e['episode_index'] // info['chunks_size'], video_key=keys[c], episode_index=e['episode_index'])) for c in cameras] for e in episodes]
        starts = np.zeros((len(episodes), len(cameras)), np.float32)
        state, action = np.concatenate([stack(t, state_keys) for t in tables]), np.concatenate([stack(t, action_keys) for t in tables])
    else:
        episodes = pq.read_table(sorted((root / 'meta/episodes').rglob('*.parquet'))[0].parent.parent).to_pandas().sort_values('episode_index')
        table = pq.read_table(root / 'data', columns=columns + ['index']).sort_by('index')
        names = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
        names = dict(zip(names['task_index'], names['task'] if 'task' in names else names.index))
        lengths = (episodes['dataset_to_index'] - episodes['dataset_from_index']).tolist()
        first = table['task_index'].to_numpy()[episodes['dataset_from_index'].to_numpy()]
        tasks = [str(names[i]) for i in first]
        files = [[str(root / info['video_path'].format(video_key=keys[c], chunk_index=e[f'videos/{keys[c]}/chunk_index'], file_index=e[f'videos/{keys[c]}/file_index'])) for c in cameras] for _, e in episodes.iterrows()]
        starts = np.stack([episodes[f'videos/{keys[c]}/from_timestamp'].to_numpy() for c in cameras], 1).astype(np.float32)
        state, action = stack(table, state_keys), stack(table, action_keys)
    return {'state': state, 'action': action, 'lengths': lengths, 'tasks': tasks, 'files': files, 'starts': starts, 'fps': info['fps']}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[3:] or EMBODIMENTS:
        pattern, cameras, state_keys, action_keys, stride = EMBODIMENTS[name]
        roots = sorted(p for p in root.glob(pattern) if (p / 'meta/info.json').exists())
        with ProcessPoolExecutor(32) as pool:
            parts = [p for p in tqdm(pool.map(index_dataset, [(r, cameras, state_keys, action_keys) for r in roots]), total=len(roots), desc=name) if p]
        if not parts:
            print(f'{name}: no datasets under {root / pattern}')
            continue
        state, action = (np.concatenate([p[k] for p in parts]) for k in ('state', 'action'))
        lengths = np.concatenate([p['lengths'] for p in parts])
        files = sorted({f for p in parts for row in p['files'] for f in row})
        lookup = {f: i for i, f in enumerate(files)}
        (out / name).mkdir(exist_ok=True)
        np.save(out / name / 'state.npy', state)
        np.save(out / name / 'action.npy', action)
        np.savez(
            out / name / 'meta.npz', first=np.concatenate([[0], np.cumsum(lengths)]),
            tasks=np.array([t for p in parts for t in p['tasks']]), files=np.array(files),
            video=np.array([[lookup[f] for f in row] for p in parts for row in p['files']]),
            start=np.concatenate([p['starts'] for p in parts]), fps=np.concatenate([np.full(len(p['lengths']), p['fps']) for p in parts]),
            state_range=np.quantile(state[::max(1, len(state) // 1_000_000)], [0.01, 0.99], axis=0) if state.shape[1] else np.zeros((2, 0)),
            action_range=np.quantile(action[::max(1, len(action) // 1_000_000)], [0.01, 0.99], axis=0), stride=stride,
        )
        print(f'{name}: {len(parts)}/{len(roots)} datasets, {len(lengths)} episodes, {len(state)} frames ({(lengths / np.concatenate([np.full(len(p["lengths"]), p["fps"]) for p in parts])).sum() / 3600:.1f} h), '
              f'state {state.shape[1]}, action {action.shape[1]}, {len(set(t for p in parts for t in p["tasks"]))} instructions')


class Mixture(torch.utils.data.IterableDataset):
    """Endless stream of samples: an embodiment by weight, then one of its frames uniformly."""

    def __init__(self, index_dir: Path, weights: dict[str, float], seed: int = 0, split: str = 'train', image_area: int = 256 * 256) -> None:
        self.names = list(weights)
        self.weights = np.array([weights[n] for n in self.names], dtype=np.float64) / sum(weights.values())
        # the per-frame tables stay on disk (memory-mapped), shared by the data workers
        self.index = {n: dict(np.load(Path(index_dir) / n / 'meta.npz')) | {k: np.load(Path(index_dir) / n / f'{k}.npy', mmap_mode='r') for k in ('state', 'action')}
                      for n in self.names}
        self.seed, self.split, self.image_area = seed, split, image_area

    def frame(self, path: str, time: float) -> Image.Image:
        """The video frame at a time, resized to about image_area with sides that are multiples of 32."""
        with av.open(path) as container:
            stream = container.streams.video[0]
            container.seek(int(time / stream.time_base), stream=stream)
            for decoded in container.decode(stream):
                if decoded.time >= time - 1e-4:
                    break
        scale = (self.image_area / (decoded.width * decoded.height)) ** 0.5
        size = [max(32, round(s * scale / 32) * 32) for s in (decoded.width, decoded.height)]
        return decoded.to_image().resize(size, Image.BILINEAR)

    def sample(self, name: str, episode: int, t: int) -> dict:
        index = self.index[name]
        first, last, stride = index['first'][episode], index['first'][episode + 1] - 1, int(index['stride'])
        steps = first + t + stride * np.arange(HORIZON)
        scale = lambda x, r: np.clip(2 * (x - r[0]) / np.maximum(r[1] - r[0], 1e-6) - 1, -5, 5).astype(np.float32)
        return {
            'embodiment': name, 'text': str(index['tasks'][episode]),
            'images': [self.frame(str(index['files'][f]), float(s) + t / float(index['fps'][episode])) for f, s in zip(index['video'][episode], index['start'][episode])],
            'state': scale(index['state'][first + t], index['state_range']),
            'action': scale(index['action'][np.minimum(steps, last)], index['action_range']), 'valid': steps <= last,
        }

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0])
        # every HOLDOUT-th episode is held out for validation
        chosen = {n: np.flatnonzero((np.arange(len(self.index[n]['first']) - 1) % HOLDOUT == 0) == (self.split == 'val')) for n in self.names}
        share = {n: np.diff(self.index[n]['first'])[chosen[n]] / np.diff(self.index[n]['first'])[chosen[n]].sum() for n in self.names}
        while True:
            name = self.names[rng.choice(len(self.names), p=self.weights)]
            first = self.index[name]['first']
            # uniform over frames: an episode in proportion to its length, then a frame
            episode = int(chosen[name][rng.choice(len(chosen[name]), p=share[name])])
            yield self.sample(name, episode, int(rng.integers(first[episode + 1] - first[episode])))


if __name__ == '__main__':
    main()

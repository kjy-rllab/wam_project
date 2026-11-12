"""Render RoboCasa scenes with ground truth for probing a VLM.

Each scene is one reset of a task in a pretrain kitchen (as the DreamZero sim eval) followed by a short random arm
motion. <out>/<task>_<seed>.npz holds the three cameras (left, right, wrist) as the policy sees them (rgb_256) and at
768 (rgb), with real depth in meters, an entity mask (index into meta['entities']) and the camera matrices at 768:
K (intrinsics) and E (camera pose in the world, OpenCV axes). meta (JSON) has the instruction, the entities (objects
with their category, fixtures with their class, robot parts), the fixtures the task refers to, object positions and
the robot state.

    python qwen_vla/collect_scenes.py --output-dir ~/project/WAM/results/qwen38_vla/scenes --seeds 5 --shard 0 2

Run in the robocasa environment.
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

os.environ['MUJOCO_GL'] = os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np

CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
# The 65 atomic tasks of the RoboCasa365 pretrain human data
TASKS = """AdjustToasterOvenTemperature AdjustWaterTemperature CheesyBread CloseBlenderLid CloseCabinet CloseDishwasher
CloseDrawer CloseElectricKettleLid CloseFridge CloseFridgeDrawer CloseMicrowave CloseOven CloseStandMixerHead
CloseToasterOvenDoor CoffeeServeMug CoffeeSetupMug LowerHeat MakeIcedCoffee NavigateKitchen OpenBlenderLid OpenCabinet
OpenDishwasher OpenDrawer OpenElectricKettleLid OpenFridge OpenFridgeDrawer OpenMicrowave OpenOven OpenStandMixerHead
OpenToasterOvenDoor PackDessert PickPlaceCabinetToCounter PickPlaceCounterToBlender PickPlaceCounterToCabinet
PickPlaceCounterToDrawer PickPlaceCounterToMicrowave PickPlaceCounterToOven PickPlaceCounterToSink
PickPlaceCounterToStandMixer PickPlaceCounterToStove PickPlaceCounterToToasterOven PickPlaceDrawerToCounter
PickPlaceFridgeDrawerToShelf PickPlaceFridgeShelfToDrawer PickPlaceMicrowaveToCounter PickPlaceSinkToCounter
PickPlaceStoveToCounter PickPlaceToasterOvenToCounter PickPlaceToasterToCounter PreheatOven SlideDishwasherRack
SlideOvenRack SlideToasterOvenRack StartCoffeeMachine TurnOffMicrowave TurnOffSinkFaucet TurnOffStove TurnOnBlender
TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet TurnOnStove TurnOnToaster TurnOnToasterOven TurnSinkSpout""".split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    parser.add_argument('--seeds', type=int, default=5, help='scenes per task')
    parser.add_argument('--size', type=int, default=768, help='side of the ground-truth renders')
    parser.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'), help='do every Nth scene')
    parser.add_argument('--workspace', type=Path, default=Path.home() / 'robocasa_workspace')
    args = parser.parse_args()

    sys.path[:0] = [str(args.workspace / 'robosuite'), str(args.workspace / 'robocasa')]
    import robocasa  # noqa: F401 -- registers the tasks
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.utils import camera_utils

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [(task, seed) for seed in range(args.seeds) for task in args.tasks]
    for task, seed in scenes[args.shard[0]::args.shard[1]]:
        path = args.output_dir / f'{task}_{seed}.npz'
        if path.exists():
            continue
        try:
            env = robosuite.make(
                env_name=task, robots='PandaOmron', controller_configs=load_composite_controller_config(robot='PandaOmron'),
                camera_names=CAMERAS, camera_widths=256, camera_heights=256, has_renderer=False,
                has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=True, seed=seed, ignore_done=True,
                control_freq=20, obj_instance_split='pretrain', layout_ids=-2, style_ids=-2,
            )
            obs = env.reset()
        except Exception as e:
            print(f'{task} seed {seed}: FAILED {type(e).__name__}: {str(e)[:200]}', flush=True)
            continue

        # A short straight arm motion (arm mode, base still), so the gripper is not always at its home pose
        rng = np.random.default_rng([seed, zlib.crc32(task.encode())])
        action = np.zeros(12)
        action[:3], action[6], action[11] = rng.uniform(-1, 1, 3), rng.choice([-1, 1]), -1
        steps = int(rng.integers(0, 40))
        for _ in range(steps):
            obs = env.step(action)[0]

        sim, model, ep = env.sim, env.sim.model, env.get_ep_meta()
        cats = {cfg['name']: str(cfg['info']['cat']) for cfg in ep['object_cfgs']}
        entities, index = [{'name': 'none', 'kind': 'none', 'cat': 'none'}], {}
        rgb, depth, mask = [], [], []
        for camera in CAMERAS:  # renders are bottom-up (OpenGL)
            image, d = sim.render(camera_name=camera, width=args.size, height=args.size, depth=True)
            geoms = sim.render(camera_name=camera, width=args.size, height=args.size, segmentation=True)[::-1, :, 1]
            for geom in np.unique(geoms[geoms >= 0]):
                chain, body = [], model.geom_bodyid[geom]
                while body > 0:
                    chain.append(model.body_id2name(body))
                    body = model.body_parentid[body]
                root = chain[-1].removesuffix('_main') if chain else 'world'
                if root in cats:
                    entity = {'name': root, 'kind': 'object', 'cat': cats[root]}
                elif root in env.fixtures:
                    entity = {'name': root, 'kind': 'fixture', 'cat': type(env.fixtures[root]).__name__}
                elif root.startswith('robot0'):
                    part = 'gripper' if chain[0].startswith('gripper0') else 'base' if 'base' in chain[0] else 'arm'
                    entity = {'name': part, 'kind': 'robot', 'cat': part}
                else:
                    entity = {'name': root, 'kind': 'other', 'cat': root}
                if entity['name'] not in [e['name'] for e in entities]:
                    entities.append(entity)
                index[geom] = [e['name'] for e in entities].index(entity['name'])
            rgb.append(image[::-1])
            depth.append(camera_utils.get_real_depth_map(sim, d)[::-1])
            mask.append(np.vectorize(lambda g: index.get(g, 0), otypes=[np.uint8])(geoms))
        meta = {
            'task': task, 'seed': seed, 'lang': ep['lang'], 'layout_id': ep['layout_id'], 'style_id': ep['style_id'],
            'steps': steps, 'entities': entities, 'fixture_refs': ep['fixture_refs'],
            'objects': {name: {'cat': cats[name], 'pos': sim.data.body_xpos[env.obj_body_id[name]].tolist()}
                        for name in env.objects},
            'robot': {key: np.asarray(obs[f'robot0_{key}']).tolist()
                      for key in ('eef_pos', 'eef_quat', 'base_pos', 'base_quat', 'base_to_eef_pos', 'gripper_qpos')},
        }
        np.savez_compressed(
            path, rgb=np.stack(rgb), rgb_256=np.stack([obs[f'{camera}_image'][::-1] for camera in CAMERAS]),
            depth=np.stack(depth).astype(np.float16), mask=np.stack(mask),
            K=np.stack([camera_utils.get_camera_intrinsic_matrix(sim, c, args.size, args.size) for c in CAMERAS]),
            E=np.stack([camera_utils.get_camera_extrinsic_matrix(sim, c) for c in CAMERAS]), meta=json.dumps(meta),
        )
        print(f'{task} seed {seed}: {len(entities)} entities, "{ep["lang"]}"', flush=True)
        env.close()


if __name__ == '__main__':
    main()

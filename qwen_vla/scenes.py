"""RoboCasa probe scenes (collect_scenes.py): the things in a camera image that a phrase names, with ground truth."""
import json
import re
from pathlib import Path

import numpy as np

CAMERAS = ['left', 'right', 'wrist']
PHRASES = {  # object categories whose name is not the phrase; the rest read as written
    'bread_flat': 'flatbread', 'kettle_non_electric': 'kettle', 'jug_wide_opening': 'jug', 'boxed_drink': 'drink box',
    'oil_and_vinegar_bottle': 'oil bottle',
    # fixture classes that are single appliances
    'Microwave': 'microwave', 'Dishwasher': 'dishwasher', 'ToasterOven': 'toaster oven', 'Toaster': 'toaster',
    'CoffeeMachine': 'coffee machine', 'Stovetop': 'stove', 'Stove': 'stove', 'Sink': 'sink', 'Oven': 'oven',
    'FridgeFrenchDoor': 'refrigerator', 'FridgeBottomFreezer': 'refrigerator', 'Blender': 'blender',
    'StandMixer': 'stand mixer', 'ElectricKettle': 'electric kettle',
}


def load(path: Path) -> dict:
    data = np.load(path)
    return {'path': Path(path).stem, **{k: data[k] for k in data.files if k != 'meta'}, **json.loads(str(data['meta']))}


def targets(scene: dict, camera: int, min_area: float = 0.001) -> list[dict]:
    """Objects, appliances and the gripper that are visible in one camera image and that their phrase names uniquely.
    Each has its phrase, role (object, appliance, gripper), mask and tight box (x1, y1, x2, y2 in 0-1000, as Qwen
    writes boxes), and, for objects, its position in the world."""
    mask, found = scene['mask'][camera], []
    for index, entity in enumerate(scene['entities']):
        if entity['kind'] == 'object':
            phrase, role = PHRASES.get(entity['cat'], entity['cat'].replace('_', ' ')), 'object'
        elif entity['kind'] == 'fixture' and entity['cat'] in PHRASES:
            phrase, role = PHRASES[entity['cat']], 'appliance'
        elif entity['name'] == 'gripper':
            phrase, role = 'robot gripper', 'gripper'
        else:
            continue
        m = mask == index
        if m.mean() < min_area:
            continue
        ys, xs = np.nonzero(m)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) * 1000 / mask.shape[0]
        found.append({'name': entity['name'], 'phrase': phrase, 'role': role, 'mask': m, 'box': box,
                      'pos': scene['objects'].get(entity['name'], {}).get('pos')})
    phrases = [t['phrase'] for t in found]
    return [t for t in found if phrases.count(t['phrase']) == 1]


def named_objects(scene: dict) -> list[int]:
    """Entities that are objects whose category the instruction names (its phrase, or the category's first word)."""
    words = lambda cat: {PHRASES.get(cat, cat.replace('_', ' ')), cat.replace('_', ' '), cat.split('_')[0]}
    return [i for i, e in enumerate(scene['entities']) if e['kind'] == 'object'
            and any(re.search(rf'\b{re.escape(w)}s?\b', scene['lang'].lower()) for w in words(e['cat']))]


def cells(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Mean of a pixel map over every image-token cell, [rows, cols]."""
    h, w = mask.shape[0] // rows, mask.shape[1] // cols
    return mask[:rows * h, :cols * w].reshape(rows, h, cols, w).mean((1, 3))

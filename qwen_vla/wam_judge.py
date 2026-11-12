"""Can the VLM judge the futures a world-action model imagines? (DreamZero context-noise samples)

A sample folder of dreamzero/eval_utils/robocasa_context_noise_samples.py has, per context and seed, the video DreamZero
predicted at every context noise level (level 0: the clean observation, level 1: no visual context) and the real
future. The VLM sees four frames of one such video (the three cameras side by side: the observation, then 0.4, 0.8
and 1.2 s later) and answers two yes/no questions, scored by logit(yes) - logit(no):

    scene     do the later frames show the same scene as the first frame?
    progress  given the instruction, is the robot making progress on the task?

Writes judge.npz: scores [contexts, seeds, rows, questions] (row 0: the real future, then the levels) and the action
error of every sample (mean absolute difference from the demonstration's actions, [contexts, seeds, levels]).

    python -m qwen_vla.wam_judge --model <checkpoint> --samples <sample folder> --output <dir>
"""
import argparse
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qwen_vla import vlm

QUESTIONS = {
    'scene': 'These are four frames of a robot video in time order; each frame shows three cameras side by side. '
             'Do the later frames show the same scene as the first frame? Answer yes or no.',
    'progress': 'These are four frames of a robot video in time order; each frame shows three cameras side by side. '
                'The task is: "{}" Is the robot making progress on the task? Answer yes or no.',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='use only the first N contexts')
    args = parser.parse_args()

    model, processor = vlm.load(args.model)
    first = lambda word: processor.tokenizer(word, add_special_tokens=False).input_ids[0]
    yes, no = ([first(w) for w in words] for words in (('yes', 'Yes'), ('no', 'No')))
    paths = sorted((args.samples / 'samples').glob('*.npz'))[:args.limit]
    scores, errors = [], []
    for path in tqdm(paths):
        sample = np.load(path)
        errors.append(np.abs(sample['actions'] - sample['actions_gt']).mean((2, 3)))  # [seeds, levels]
        rows = []
        for video in sorted((args.samples / 'videos').glob(f'{path.stem}_seed*.mp4'), key=lambda p: int(p.stem.split('seed')[1])):
            with av.open(str(video)) as container:
                frames = np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])[[0, 5, 10, 16]]
            strips = frames.reshape(4, -1, 224, *frames.shape[2:]).transpose(1, 0, 2, 3, 4)  # [rows, 4 frames, 224, 672, 3]
            prompts = [([Image.fromarray(f) for f in strip], q.format(str(sample['language']))) for strip in strips for q in QUESTIONS.values()]
            inputs = vlm.chat(processor, prompts).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1].float()
            rows.append((logits[:, yes].logsumexp(-1) - logits[:, no].logsumexp(-1)).reshape(len(strips), len(QUESTIONS)).cpu().numpy())
        scores.append(np.stack(rows))
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = min(len(s) for s in scores)
    scores, errors = np.stack([s[:seeds] for s in scores]), np.stack([e[:seeds] for e in errors])
    np.savez(args.output / 'judge.npz', scores=scores, errors=errors, contexts=[p.stem for p in paths], questions=list(QUESTIONS),
             levels=np.load(paths[0])['levels'])
    for q, name in enumerate(QUESTIONS):
        print(f'{name:9} real {scores[:, :, 0, q].mean():+.2f} | by level ' + ' '.join(f'{v:+.2f}' for v in scores[:, :, 1:, q].mean((0, 1))))


if __name__ == '__main__':
    main()

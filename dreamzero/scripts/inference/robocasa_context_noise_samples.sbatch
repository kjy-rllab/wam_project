#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4  # CPUs and memory follow the GPU count (hangang rule for shared nodes)
#SBATCH --time=04:00:00  # the defaults (48 contexts x 9 levels x 5 seeds) take about 1.5 h on 4 GPUs, --y-tail swap --seeds 2 about 50 min
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: video and action samples of a checkpoint at fixed context noise levels
# (eval_utils/robocasa_context_noise_samples.py). Submit from the dreamzero folder on /scratch, with the global steps
# the run's jobs started from:
#   $ sbatch scripts/inference/robocasa_context_noise_samples.sbatch <checkpoint dir> --resume-steps 0 9000 18000
#   $ sbatch scripts/inference/robocasa_context_noise_samples.sbatch <checkpoint dir> --resume-steps 0 9000 18000 --y-tail swap --seeds 2
# A resubmission skips the contexts already saved.

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${DREAMZERO_ENV:-$WORK/envs/dreamzero}"
export PYTHONPATH=$PWD  # this checkout's groot (the env's editable install may point to another checkout)
export HF_HUB_OFFLINE=1

torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" eval_utils/robocasa_context_noise_samples.py --model-path "$1" "${@:2}"

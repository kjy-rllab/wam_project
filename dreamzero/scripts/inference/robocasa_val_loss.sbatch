#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4  # CPUs and memory follow the GPU count (hangang rule for shared nodes)
#SBATCH --time=03:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: validation loss of a checkpoint at fixed context noise levels (eval_utils/robocasa_val_loss.py).
# Submit from the dreamzero folder on /scratch, with the global steps the run's jobs started from:
#   $ sbatch scripts/inference/robocasa_val_loss.sbatch <checkpoint dir> --resume-steps 0 9000 18000

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${DREAMZERO_ENV:-$WORK/envs/dreamzero}"
export PYTHONPATH=$PWD  # this checkout's groot (the env's editable install may point to another checkout)
export HF_HUB_OFFLINE=1

torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" eval_utils/robocasa_val_loss.py --model-path "$1" "${@:2}"

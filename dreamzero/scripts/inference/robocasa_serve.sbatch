#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2  # CPUs and memory follow the GPU count (hangang rule for shared nodes)
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: serve a checkpoint to the local RoboCasa client (eval_utils/serve_robocasa.py, CFG on 2 GPUs).
# Submit from the dreamzero folder on /scratch:
#   $ sbatch scripts/inference/robocasa_serve.sbatch <checkpoint dir> [--tcont-context 0.3]
# Then on the local PC, with the node and port from the job log ("Serving ... on port"):
#   $ ssh -N -L 5555:<node>:<port> hangang.ksc.re.kr

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${DREAMZERO_ENV:-$WORK/envs/dreamzero}"
export PYTHONPATH=$PWD  # this checkout's groot (the env's editable install may point to another checkout)
export HF_HUB_OFFLINE=1

# Nodes are shared with other jobs, so neither the server port nor the torchrun port is fixed
torchrun --standalone --nproc_per_node="$SLURM_GPUS_ON_NODE" eval_utils/serve_robocasa.py \
    --model-path "$1" --port $((10000 + SLURM_JOB_ID % 50000)) "${@:2}"

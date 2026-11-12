#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: run one command of this repository on a node (4 GPUs), in the qwen environment.
#   $ sbatch [--gres=gpu:1] [--time=..] qwen_vla/scripts/job.sbatch python -m qwen_vla.wam_judge --model ...

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1 aws-ofi-nccl
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$WORK/envs/qwen"
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
(lfs find "$WORK" -print0 | xargs -0 -n 2000 touch -a -c -h) > /dev/null 2>&1 &
"$@"

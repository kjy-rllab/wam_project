#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#
# KISTI hangang: train a VLA (qwen_vla/train.py) on all GPUs of the job. Submit from the repository root on /scratch:
#   $ sbatch [--nodes=N] qwen_vla/scripts/train.sbatch <run name> --mode frozen --tap 32 --weights '{"robocasa": 1}' [train.py options]
# The run goes to $WORK/runs/qwen_vla/<run name>.

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1 aws-ofi-nccl  # aws-ofi-nccl: NCCL over Slingshot
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$WORK/envs/qwen"
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
# /scratch deletes files not read for 15 days
(lfs find "$WORK" -print0 | xargs -0 -n 2000 touch -a -c -h) > /dev/null 2>&1 &

export MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun --kill-on-bad-exit=1 bash -c 'torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=4 \
    --master_addr=$MASTER --master_port=29511 -m qwen_vla.train --vlm "${VLM:-'"$WORK"'/checkpoints/Qwen3.8-27B}" \
    --index '"$WORK"'/datasets/qwen_index --output '"$WORK"'/runs/qwen_vla/"$0" "$@"' "$@"

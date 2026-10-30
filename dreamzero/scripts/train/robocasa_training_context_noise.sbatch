#!/bin/bash
#SBATCH --job-name=dz-robocasa-cn
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
# KISTI hangang (GH200, natStr) multi-node launch of robocasa_training_context_noise.sh.
# Submit from the dreamzero folder on /scratch (jobs can only be submitted from /scratch or /appsdata):
#   $ sbatch scripts/train/robocasa_training_context_noise.sbatch [hydra overrides]
# 24h job limit: each job queues the next one, which resumes from the latest checkpoint in OUTPUT_DIR.
# Keep the node count fixed for a run (ZeRO optimizer shards are saved per GPU).

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1 aws-ofi-nccl  # aws-ofi-nccl: NCCL over Slingshot
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${DREAMZERO_ENV:-$WORK/envs/dreamzero}"

export ROBOCASA_DATA_ROOT=${ROBOCASA_DATA_ROOT:-"$WORK/datasets/robocasa/v1.0/pretrain/*/*/*/lerobot"}
export WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-$WORK/checkpoints/Wan2.1-I2V-14B-480P}
export TOKENIZER_DIR=${TOKENIZER_DIR:-$WORK/checkpoints/umt5-xxl}
export OUTPUT_DIR=${OUTPUT_DIR:-$WORK/runs/dreamzero_robocasa_context_noise}
export NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}
export HF_HUB_OFFLINE=1
export MAX_STEPS=${MAX_STEPS:-25000}  # global batch 64 on 16 nodes: 1.6M samples

# Queue the next job of the chain (at most CHAIN_JOBS jobs); it starts when this one ends
export CHAIN_JOBS=${CHAIN_JOBS:-4}
if [ "$CHAIN_JOBS" -gt 1 ] && [ ! -f "$OUTPUT_DIR/config.json" ]; then
    NEXT_JOB=$(CHAIN_JOBS=$((CHAIN_JOBS - 1)) sbatch --parsable --nodes="$SLURM_NNODES" --dependency=afterany:"$SLURM_JOB_ID" \
        scripts/train/robocasa_training_context_noise.sbatch "$@")
    echo "Queued next job: $NEXT_JOB"
fi

srun --kill-on-bad-exit=1 bash -c 'NODE_RANK=$SLURM_NODEID bash scripts/train/robocasa_training_context_noise.sh "$@"' _ "$@"
status=$?

# Training finished (final model saved in OUTPUT_DIR): the queued job is not needed
if [ -f "$OUTPUT_DIR/config.json" ] && [ -n "$NEXT_JOB" ]; then scancel "$NEXT_JOB"; fi
exit $status

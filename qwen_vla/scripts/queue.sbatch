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
# KISTI hangang: one job that runs the commands of a queue file one after another, so several tasks share one
# submission (the lab has 5 job slots). Lines can be added while the job runs; the line END ends it.
#   $ sbatch qwen_vla/scripts/queue.sbatch <queue file>
# A line is a shell command run from the repository root, e.g.
#   bash qwen_vla/scripts/train.sbatch <run name> --mode frozen --tap 32 --weights '{"robocasa": 1}'

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1 aws-ofi-nccl
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$WORK/envs/qwen"
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
(lfs find "$WORK" -print0 | xargs -0 -n 2000 touch -a -c -h) > /dev/null 2>&1 &

done=0
while true; do
    line=$(sed -n "$((done + 1))p" "$1")
    if [ -z "$line" ]; then sleep 30; continue; fi
    [ "$line" = END ] && break
    echo "=== queue $((done + 1)) start $(date +%T): $line"
    bash -c "$line"
    echo "=== queue $((done + 1)) exit $? $(date +%T)"
    done=$((done + 1))
done

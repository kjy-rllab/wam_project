#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: serve VLA checkpoints to local RoboCasa clients, one server per GPU on ports 5555, 5556, ...
#   $ sbatch qwen_vla/scripts/serve.sbatch <checkpoint dir> [<checkpoint dir> ...]   (up to 4; one may repeat)
# The log prints the ssh command that tunnels the ports to the local machine.

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$WORK/envs/qwen"
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
(lfs find "$WORK" -print0 | xargs -0 -n 2000 touch -a -c -h) > /dev/null 2>&1 &

i=0
for checkpoint in "$@"; do
    CUDA_VISIBLE_DEVICES=$i python -m qwen_vla.serve_robocasa --checkpoint "$checkpoint" --port $((5555 + i)) &
    echo "Local tunnel: ssh -N -L $((5555 + i)):$(hostname):$((5555 + i)) hangang.ksc.re.kr"
    i=$((i + 1))
    sleep 90  # one model load at a time
done
wait

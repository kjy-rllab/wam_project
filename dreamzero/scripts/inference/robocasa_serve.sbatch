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
# One server per context noise level, each on an equal share of the GPUs (eval_utils/robocasa_sim_eval.py):
#   $ SERVE_LEVELS="0 0.0625 0.125 0.25" sbatch --gres=gpu:4 --time=08:00:00 scripts/inference/robocasa_serve.sbatch <checkpoint dir>
# Then run on the local PC the tunnel command that the job log prints (ports 5555, 5556, ... go to the servers).

WORK=${WORK:-/scratch/t526a02/jooyoung}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "${DREAMZERO_ENV:-$WORK/envs/dreamzero}"
export PYTHONPATH=$PWD  # this checkout's groot (the env's editable install may point to another checkout)
export HF_HUB_OFFLINE=1

# Nodes are shared with other jobs, so neither the server ports nor the torchrun ports are fixed
levels=(${SERVE_LEVELS:-none})
gpus=(${CUDA_VISIBLE_DEVICES//,/ })
[ ${#gpus[@]} -eq 0 ] && gpus=($(seq 0 $((SLURM_GPUS_ON_NODE - 1))))
n=$((${#gpus[@]} / ${#levels[@]}))
port=$((10000 + SLURM_JOB_ID % 50000))
for i in "${!levels[@]}"; do
    level=()
    [ "${levels[i]}" != none ] && level=(--tcont-context "${levels[i]}")
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpus[*]:i*n:n}") torchrun --standalone --nproc_per_node=$n \
        eval_utils/serve_robocasa.py --model-path "$1" --port $((port + i)) "${level[@]}" "${@:2}" &
    # The next server starts once this one listens: a loading rank holds about 120 GB of CPU memory, and on GH200 the
    # CPU memory spills into GPU memory once full
    while kill -0 $! 2>/dev/null && ! (exec 3<>/dev/tcp/127.0.0.1/$((port + i))) 2>/dev/null; do sleep 10; done
    tunnel+=" -L $((5555 + i)):$SLURMD_NODENAME:$((port + i))"
    servers+=" ${levels[i]}=tcp://127.0.0.1:$((5555 + i))"
done
echo "Local tunnel: ssh -N$tunnel hangang.ksc.re.kr"
[ -n "$SERVE_LEVELS" ] && echo "Local eval: python eval_utils/robocasa_sim_eval.py --servers$servers --output-dir <dir>"
wait

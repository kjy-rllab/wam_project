#!/bin/bash
#SBATCH --job-name=jm
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00  # Qwen3.8-27B on 325 scenes: about 3 h
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=%x-%j.out
#
# KISTI hangang: every probe of a VLM on the RoboCasa probe scenes (behavior, attention, features and linear probes),
# four shards at a time, one per GPU. Submit from the repository root on /scratch:
#   $ sbatch qwen_vla/scripts/probe.sbatch <model dir> <output dir> [stage ...]
# Stages (default all): behavior attention features. A resubmission redoes the stages it is given.

WORK=${WORK:-/scratch/t526a02/jooyoung}
SCENES=${SCENES:-$WORK/datasets/qwen_probe/scenes}
module load miniconda3/py314_26.7.1-1
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$WORK/envs/qwen"
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
# /scratch deletes files not read for 15 days
(lfs find "$WORK" -print0 | xargs -0 -n 2000 touch -a -c -h) > /dev/null 2>&1 &

MODEL=$1 OUT=$2
STAGES=${*:3}
STAGES=${STAGES:-behavior attention features}
mkdir -p "$OUT"

shards() {  # run "$@" as four shards, one per GPU; the first loads the model before the others start reading it
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python -m "$@" --shard $i 4 > "$OUT/log_$(echo "$*" | tr ' /' '__' | cut -c1-80)_$i.txt" 2>&1 &
        [ $i = 0 ] && sleep 120
    done
    wait
}

for stage in $STAGES; do
    case $stage in
    behavior)
        shards qwen_vla.probe_behavior --model "$MODEL" --scenes "$SCENES" --output "$OUT" --size 768
        shards qwen_vla.probe_behavior --model "$MODEL" --scenes "$SCENES" --output "$OUT" --size 256 ;;
    attention)
        shards qwen_vla.probe_attention --model "$MODEL" --scenes "$SCENES" --output "$OUT" --mode locate --size 768
        shards qwen_vla.probe_attention --model "$MODEL" --scenes "$SCENES" --output "$OUT" --mode locate --size 256
        shards qwen_vla.probe_attention --model "$MODEL" --scenes "$SCENES" --output "$OUT" --mode policy ;;
    features)
        shards qwen_vla.probe_features --model "$MODEL" --scenes "$SCENES" --output "$OUT/features_policy" --layout policy
        shards qwen_vla.probe_features --model "$MODEL" --scenes "$SCENES" --output "$OUT/features_policy_text_first" --layout policy --text-first --step 4
        shards qwen_vla.probe_features --model "$MODEL" --scenes "$SCENES" --output "$OUT/features_single_768" --layout single --size 768
        for name in policy policy_text_first single_768; do
            python -m qwen_vla.probe_linear --features "$OUT/features_$name" --output "$OUT/probe_$name.json" > "$OUT/log_probe_$name.txt" 2>&1
        done ;;
    esac
done
echo PROBE_DONE

#!/bin/bash
# DreamZero pretraining on RoboCasa365 human-teleop data with noised context (tmrl), Wan2.1-I2V-14B-480P backbone
#
# Usage (one node):
#   ROBOCASA_DATA_ROOT='/path/to/robocasa/v1.0/pretrain/*/*/*/lerobot' bash scripts/train/robocasa_training_context_noise.sh
# Multi-node: run once per node with NNODES, NODE_RANK and MASTER_ADDR set
#   (see scripts/train/robocasa_training_context_noise.sbatch).
# Extra arguments are passed to Hydra as overrides, e.g. max_steps=10 save_strategy=no.
#
# Weights:
#   huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

export HYDRA_FULL_ERROR=1
DREAMZERO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============ USER CONFIGURATION ============
ROBOCASA_DATA_ROOT=${ROBOCASA_DATA_ROOT:?Set ROBOCASA_DATA_ROOT to a glob over the RoboCasa lerobot folders}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_robocasa_context_noise"}
CONTEXT_NOISE=${CONTEXT_NOISE:-true}
REPORT_TO=${REPORT_TO:-none}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_GPUS=${NUM_GPUS:-4}  # per node
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NNODES * NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-100000}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}

WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

cd "$DREAMZERO_ROOT"
python -m torch.distributed.run \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NUM_GPUS" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    groot/vla/experiment/experiment.py \
    report_to=$REPORT_TO \
    data=dreamzero/robocasa \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    optim=adamw_torch_fused \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    save_lora_only=false \
    save_strategy=steps \
    robocasa_data_root="$ROBOCASA_DATA_ROOT" \
    dit_version=$WAN21_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN21_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    ++action_head_cfg.config.context_noise=$CONTEXT_NOISE \
    "$@"

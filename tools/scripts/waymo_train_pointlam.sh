#!/bin/bash
# Usage: bash scripts/waymo_train_pointlam.sh [experiment_tag] [epochs] [gpu_list]

EXP_TAG=${1:-default}
EPOCHS=${2:-24}
GPU_LIST=${3:-0,1,2,3}

CFG="./cfgs/pointlam_models/pointlam_waymo.yaml"

NPROC=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$GPU_LIST python -m torch.distributed.launch \
    --nproc_per_node=$NPROC --master_port=29988 train.py --tcp_port 29988 --launcher pytorch \
    --cfg_file "$CFG" \
    --extra_tag "$EXP_TAG" \
    --batch_size 16 --epochs "$EPOCHS" --max_ckpt_save_num 12 --workers 8 --sync_bn

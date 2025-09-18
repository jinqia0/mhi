#!/bin/bash

# 用法提示
if [ -z "$1" ]; then
  echo "用法: $0 <csv_path>"
  exit 1
fi

CSV_PATH="$1"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 可根据实际 GPU 数量调整
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 创建日志文件
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/logs/pllava/$timestamp.log"

cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi             

# ========== 模型参数 ==========
PRETRAINED_MODEL="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/MODELS/pllava-13b"
WEIGHT_DIR="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/MODELS/pllava-13b"
NUM_FRAMES=4                                      
BATCH_SIZE=16
POOLING_SHPAE="4-12-12" # 4-12-12   
NUM_WORKERS=16                             

/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/torchrun \
    --nproc_per_node=4 \
    --standalone /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/caption_pllava.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --csv_path $CSV_PATH \
    --num_frames $NUM_FRAMES \
    --batch_size $BATCH_SIZE \
    --pooling_shape $POOLING_SHPAE \
    --num_workers $NUM_WORKERS \
    --use_lora --weight_dir $WEIGHT_DIR \
    2>&1 | tee -a $log_file


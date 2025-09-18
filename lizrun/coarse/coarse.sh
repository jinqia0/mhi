#!/bin/bash

# 用法提示
if [ -z "$1" ]; then
  echo "用法: $0 <csv_path>"
  exit 1
fi

CSV_PATH="$1"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 可根据实际 GPU 数量调整

cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/

# 启动 Python 脚本
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/python /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/filters/coarse/coarse.py \
    --csv_path "$CSV_PATH" \
    --batch_size 8 \
    --conf_thresh 0.6 \
    --iou_thresh 0.45 \
    --pool_size_per_gpu 8 \
    --model_path "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/weights/yolo/yolo11m.pt"

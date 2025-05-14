#!/bin/bash

# 用法提示
if [ -z "$1" ]; then
  echo "用法: $0 <csv_path>"
  exit 1
fi

CSV_PATH="$1"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 可根据实际 GPU 数量调整

# 创建日志文件
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/logs/ocr/$timestamp.log"

cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora

# 启动 Python 脚本
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/opensora/bin/torchrun \
    --nproc_per_node 4 --standalone \
    /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/tools/scoring/ocr/inference.py \
    --num_workers 32 --bs 16 \
    $CSV_PATH 2>&1 | tee -a $log_file

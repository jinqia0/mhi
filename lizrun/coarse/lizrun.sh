#!/bin/bash

# 参数检查
if [ "$#" -ne 2 ]; then
  echo "用法: $0 <job_name> <csv_file_path>"
  exit 1
fi

JOB_NAME="$1"
CSV_PATH="$2"

# 启动 lizrun 任务
lizrun start -j "$JOB_NAME" \
  -n 1 \
  -g 2 \
  -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
  -c "bash /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/lizrun/coarse/coarse.sh $CSV_PATH" \
  -p dhg

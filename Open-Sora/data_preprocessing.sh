#!/bin/bash

# 基本配置
PROJECT_DIR="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora"
DATASET="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/test/test_100.csv"
NUM_WORKERS=4  # 工作线程数
NUM_GPUS=4  # GPU 数量
LOG_DIR="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/logs/opensora"  # 日志输出目录

# 创建日志目录（如果不存在）
mkdir -p $LOG_DIR

# 定义任务函数
run_task() {
    local script=$1
    local input_file=$2
    local output_file=$3
    local batch_size=$4
    local log_file=$5

    echo "Running $script with input: $input_file, output: $output_file, batch size: $batch_size" | tee -a $log_file
    torchrun --nproc_per_node $NUM_GPUS --standalone \
        $script \
        --num_workers $NUM_WORKERS --bs $batch_size \
        $input_file 2>&1 | tee -a $log_file
}

# 创建日志文件
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$LOG_DIR/$timestamp.log"

echo "Logging to $log_file"

# OCR 阶段
ocr_input_file=$DATASET
ocr_output_file="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/test/test_100_ocr.csv"
run_task "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/tools/scoring/ocr/inference.py" \
    $ocr_input_file $ocr_output_file 4 $log_file

# Optical Flow 阶段
flow_input_file=$ocr_output_file
flow_output_file="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/test/test_100_ocr_flow.csv"
run_task "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/tools/scoring/optical_flow/inference.py" \
    $flow_input_file $flow_output_file 1 $log_file

# Zoom 检测阶段
zoom_input_file=$flow_output_file
zoom_output_file="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/test/test_100_ocr_flow_zoom.csv"
run_task "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/tools/scoring/optical_flow/detect_zoom.py" \
    $zoom_input_file $zoom_output_file 1 $log_file

echo "All tasks completed. Log file saved to $log_file"

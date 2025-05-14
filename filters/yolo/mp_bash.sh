#!/bin/bash

# 设置要处理的CSV文件列表
CSV_DIR="/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/csv_1M"
csv_files=($(ls ${CSV_DIR}/*.csv))

# 启动8个进程，每个进程绑定到不同的GPU，并将日志输出到不同的文件
for i in {0..7}
do
    if [ $i -lt ${#csv_files[@]} ]; then
        echo "正在处理 ${csv_files[$i]} 使用 GPU $i"
        
        # 定义日志文件名
        log_file="/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/logs/yolo/log_gpu${i}_$(basename ${csv_files[$i]} .csv).log"
        
        # 启动Python脚本，指定GPU和CSV文件路径，并将日志输出到对应文件
        /mnt/pfs-mc0p4k/cvg/team/jinqiao/envs/mhi/bin/python /mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/filters/yolo/yolo_single.py $i ${csv_files[$i]} > ${log_file} 2>&1 &
    fi
done

# 等待所有进程完成
wait

echo "所有文件处理完成"

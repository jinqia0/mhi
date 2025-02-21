#!/bin/bash
cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi  # 进入目标目录
export CUDA_VISIBLE_DEVICES=$1  # 指定使用的 GPU
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/python main.py  # 运行 Python 脚本

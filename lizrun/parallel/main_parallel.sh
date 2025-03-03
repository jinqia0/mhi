#!/bin/bash
cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1

# 配置NCCL参数
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
# export CUDA_LAUNCH_BLOCKING=1  # 调试用，正式运行可移除

/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  --max_restarts=0 \
  scripts/llama/main_parallel.py 2>&1 | tee -a log/run.log

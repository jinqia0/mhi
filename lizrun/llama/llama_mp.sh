#!/bin/bash
cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi  # 进入目标目录
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  llama_mp.py

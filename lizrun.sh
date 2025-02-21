#!/bin/bash
lizrun start -j mhi$1 \
-n 1 \
-g 1 \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/lizrun/main_lizrun.sh $2" \
-p dhg
# -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \  # 镜像

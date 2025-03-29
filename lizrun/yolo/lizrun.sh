#!/bin/bash
lizrun start -j $1 \
-n 1 \
-g 4 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/lizrun/yolo/yolo.sh" \
-p dhg

#!/bin/bash
lizrun start -j $1 \
-n 1 \
-g 2 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/lizrun/pllava/pllava.sh" \
-p dhg

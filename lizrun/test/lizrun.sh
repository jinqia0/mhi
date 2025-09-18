#!/bin/bash
lizrun start -j $1 \
-n 1 \
-g 1 \
-i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-lizr-nccl \
-c "bash lizrun/test/test.sh" \
-p dhg

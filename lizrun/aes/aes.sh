#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1

cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi

echo "Start Running AES Inference"
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/aes/bin/python /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/filters/aesthetic/aes.py
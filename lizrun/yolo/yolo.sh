#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi

echo "Start Running YOLO Inference"
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/mhi/bin/python /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/filters/yolo/yolo_mp.py >> logs/yolo_lizrun_61.log
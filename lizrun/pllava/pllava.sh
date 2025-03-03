#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True              

# ========== 模型参数 ==========
PRETRAINED_MODEL="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/MODELS/pllava-13b"
CSV_PATH="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/internvid/internvid_00_100.csv" 
NUM_FRAMES=8                                      
BATCH_SIZE=8                                

/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/pllava/bin/torchrun \
    --nproc_per_node=2 \
    --standalone /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/caption_pllava.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --use_lora \
    --lora_alpha 4 \
    --weight_dir "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/MODELS/pllava-13b" \
    --csv_path $CSV_PATH \
    --num_frames $NUM_FRAMES \
    --batch_size $BATCH_SIZE \
    --pooling_shape "4-12-12"


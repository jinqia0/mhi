#!/bin/bash
# Usage:
#    Set environment variables HF_USERNAME and HF_TOKEN, then run:
#    ./download_hfd.sh

# Set the parameters
model_id="jinqia0/mhi"
hf_username="${HF_USERNAME}"
hf_token="${HF_TOKEN}"
tool="aria2c"
threads="10"
concurrent_file='1'
dir='huggingface_datasets/mhi'
# exclude_pattern="*.safetensors"

# Run the download command
bash download_huggingface/hfd.sh $model_id \
    --hf_username $hf_username \
    --hf_token $hf_token \
    --tool $tool \
    -x $threads \
    -j $concurrent_file \
    --local-dir $dir \
    --dataset
    # --exclude $exclude_pattern

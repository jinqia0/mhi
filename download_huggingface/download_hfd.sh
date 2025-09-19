#!/bin/bash
# Usage:
#    Set environment variables HF_USERNAME and HF_TOKEN, then run:
#    ./download_hfd.sh

# Set the parameters
model_id="THUDM/CogVideoX-5b"
exclude_pattern="*.safetensors"
hf_username="${HF_USERNAME}"
hf_token="${HF_TOKEN}"
tool="aria2c"
threads="10"
concurrent_file='1'
dir='/data/Pretrained_models'

# Check if hfd.sh exists
if [ ! -f "hfd.sh" ]; then
    echo "Error: hfd.sh not found. Please make sure it exists in the same directory."
    exit 1
fi

# Run the download command
bash hfd.sh $model_id \
    --hf_username $hf_username \
    --hf_token $hf_token \
    --tool $tool \
    -x $threads \
    -j $concurrent_file \
    --local-dir $dir \
    # --exclude $exclude_pattern \
    # --dataset

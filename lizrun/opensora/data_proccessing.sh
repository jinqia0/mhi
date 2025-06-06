# ROOT_VIDEO="data-test-1k"
cd /mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora
ROOT_CLIPS="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/data-test-1k/clips"
ROOT_META="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/data-test-1k/meta"

# # 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
# python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

# # 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin1.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1

# # 2.1 Detect scenes. This should output ${ROOT_META}/meta_info_fmin1_timestamp.csv
# python -m tools.scene_cut.scene_detect ${ROOT_META}/meta_info_fmin1.csv

# # 2.2 Cut video into clips based on scenes. This should produce video clips under ${ROOT_CLIPS}
# python -m tools.scene_cut.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}

# # 2.3 Create a meta file for video clips. This should output ${ROOT_META}/meta_clips.csv
# python -m tools.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv

# # 2.4 Get clips information and remove broken ones. This should output ${ROOT_META}/meta_clips_info_fmin1.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1

# 3.1 Predict aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes.csv
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/opensora/bin/torchrun --nproc_per_node 1 -m tools.scoring.aesthetic.inference \
  ${ROOT_META}/meta_clips_info_fmin1.csv \
  --bs 1024 \
  --num_workers 16

# 3.2 Filter by aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/opensora/bin/python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes.csv --aesmin 5

# # 4.1 Generate caption. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0_caption_part*.csv
# torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llava \
#   ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv \
#   --dp-size 8 \
#   --tp-size 1 \
#   --model-path /path/to/llava-v1.6-mistral-7b \
#   --prompt video

# # 4.2 Merge caption results. This should output ${ROOT_META}/meta_clips_caption.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0_caption_part*.csv --output ${ROOT_META}/meta_clips_caption.csv

# # 4.3 Clean caption. This should output ${ROOT_META}/meta_clips_caption_cleaned.csv
# python -m tools.datasets.datautil \
#   ${ROOT_META}/meta_clips_caption.csv \
#   --clean-caption \
#   --refine-llm-caption \
#   --remove-empty-caption \
#   --output ${ROOT_META}/meta_clips_caption_cleaned.csv

# # 4.4 Optionally generate tags (e.g., objects) based on the captions. This should output your_output_prefix_{key}.csv
# torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llama3 ${ROOT_META}/meta_clips_caption_cleaned.csv --key objects --output_prefix your_output_prefix

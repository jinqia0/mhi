import sys
import os
from pathlib import Path

datasets_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/opensora/datasets"
sys.path.append(datasets_dir)
from read_video import read_video_av
sys.path.remove(datasets_dir)

import logging
import numpy as np
import pandas as pd
import torch
import torchvision
import transformers
from PIL import Image
from tasks.eval.eval_utils import Conversation
from tasks.eval.model_utils import load_pllava
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

from caption_pllava import pllava_answer, get_index, load_video, collate_fn, parse_args, infer

conv_template = Conversation(
    system="Please describe the content of this video in as much detail as possible. Please describe the content of the video and the changes that occur, in chronological order. \
            The description should be useful for AI to re-generate the video. The description should not be less than six sentences. Here is one example of good descriptions: \
                1. There is only one person in the video, a young girl of yellow race. She is wearing a black sling and has brown curly hair that falls on her shoulders and back. \
                She has a fair figure, big eyes, delicate skin, deep and bright eyes, and blood-red lips. She looks sexy and has delicate makeup. \
                Her expression is calm at first, but then she smiles seductively at the camera. \
                The background is the blurred outline of city buildings and windows, through which you can see the scenery outside.",  # 保持原有prompt不变
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token="<image>",
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672

# 保持原有的pllava_answer、get_index、load_video、collate_fn函数不变

class CSVDataset(Dataset):
    def __init__(self, csv_path, num_frames):
        self.df = pd.read_csv(csv_path)
        self.data_list = self.df.path.tolist()
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError
        try:
            video = load_video(self.data_list[idx], self.num_frames, resolution=RESOLUTION)
        except:
            return None
        return video


def load_model_and_dataset(args, pooling_shape=(16, 12, 12)):
    # 修改后的模型加载函数
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha,
        pooling_shape=pooling_shape,
    )
    print("START HERE*****************************************************************")
    print(model.config)
    print(model)
    print("END HERE*******************************************************************")
    logger.info("done loading llava")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    dataset = CSVDataset(args.csv_path, args.num_frames)
    return model, processor, dataset

def main():
    args = parse_args()
    
    # 加载模型和数据集
    model, processor, dataset = load_model_and_dataset(args)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    total = 0
    result_list = []
    print(f"Total samples to process: {len(dataset)}")
    
    for batch in tqdm(dataloader):
        total += 1
        try:
            preds = infer(
                model,
                processor,
                batch,
                conv_mode=args.conv_mode,
                print_res=False,  # 调试时可设为True
            )
        except Exception as e:
            logger.error(f"error in {batch}: {str(e)}")
            preds = [args.error_message] * len(batch)
        result_list.extend(preds)

    # 保存结果
    df = pd.read_csv(args.csv_path)
    df["text"] = result_list
    
    # 删除失败条目（可选）
    df = df[df["text"] != args.error_message]
    
    new_csv_path = args.csv_path.replace(".csv", "_text.csv")
    df.to_csv(new_csv_path, index=False)
    print(f"Results saved to {new_csv_path}")

if __name__ == "__main__":
    main()

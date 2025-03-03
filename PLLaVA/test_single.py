import numpy as np
import logging
import pandas as pd
import torch
from PIL import Image
from tasks.eval.eval_utils import Conversation
from tasks.eval.model_utils import load_pllava
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

from caption_pllava import pllava_answer, get_index, load_video, collate_fn, parse_args, infer

import sys
import os

datasets_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/opensora/datasets"
sys.path.append(datasets_dir)
from read_video import read_video_av
sys.path.remove(datasets_dir)

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

def pllava_answer(
    conv: Conversation,
    model,
    processor,
    video_list,
    do_sample=True,
    max_new_tokens=200,
    num_beams=1,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1,
    temperature=1.0,
    print_res=False,
):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs_list = [processor(text=prompt, images=video, return_tensors="pt") for video in video_list]
    inputs_batched = dict()  # add batch dimension by cat
    for input_type in list(inputs_list[0].keys()):
        inputs_batched[input_type] = torch.cat([inputs[input_type] for inputs in inputs_list])
    inputs_batched = BatchFeature(inputs_batched, tensor_type="pt").to(model.device)

    with torch.no_grad():
        output_texts = model.generate(
            **inputs_batched,
            media_type="video",
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_texts = processor.batch_decode(
            output_texts, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    for i in range(len(output_texts)):
        if print_res:  # debug usage
            print("### PROMPTING LM WITH: ", prompt)
            print("### LM OUTPUT TEXT:  ", output_texts[i])
        if conv.roles[-1] == "<|im_start|>assistant\n":
            split_tag = "<|im_start|> assistant\n"
        else:
            split_tag = conv.roles[-1]
        output_texts[i] = output_texts[i].split(split_tag)[-1]
        ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
        output_texts[i] = output_texts[i].removesuffix(ending).strip()
        output_texts[i] = output_texts[i].replace("\n", " ")
        conv.messages[-1][1] = output_texts[i]
    return output_texts, conv

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
    return offsets

def load_video(video_path, num_frames, return_msg=False, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vframes, aframes, info = read_video_av(
        video_path,
        pts_unit="sec", 
        output_format="THWC"
    )
    total_num_frames = len(vframes)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vframes[frame_index].numpy())
        images_group.append(transforms(img))
    if return_msg:
        exit('return_msg not implemented yet')
    else:
        return images_group


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
    
    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.data_per_gpu = len(self) // world_size
        start_index = rank * self.data_per_gpu
        end_index = (rank + 1) * self.data_per_gpu if rank != world_size - 1 else len(self)
        self.data_list = self.data_list[start_index:end_index]



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

import sys
import os

datasets_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Open-Sora/opensora/datasets"
sys.path.append(datasets_dir)
from read_video import read_video_av
sys.path.remove(datasets_dir)

import itertools
import logging
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import torch
import torchvision
import transformers
from decord import VideoReader, cpu
from PIL import Image
from tasks.eval.eval_utils import Conversation
from tasks.eval.model_utils import load_pllava
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

conv_template = Conversation(
    system="Please describe the content of this video in as much detail as possible. Please describe the content of the video and the changes that occur, in chronological order. \
The description should be useful for AI to re-generate the video. The description should not be less than six sentences. Here is one example of good descriptions: \
1. There is only one person in the video, a young girl of yellow race. She is wearing a black sling and has brown curly hair that falls on her shoulders and back. \
She has a fair figure, big eyes, delicate skin, deep and bright eyes, and blood-red lips. She looks sexy and has delicate makeup. \
Her expression is calm at first, but then she smiles seductively at the camera. \
The background is the blurred outline of city buildings and windows, through which you can see the scenery outside.",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token="<image>",
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 472  #

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def pllava_answer(
        conv: Conversation,
        model,
        processor,
        video_list,
        do_sample=True,
        max_new_tokens=500,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
        print_res=False
    ):
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
    del inputs_batched
    torch.cuda.empty_cache()
    return output_texts, conv


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
    return offsets


def load_video(video_path, num_frames, return_msg=False, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vframes, _, _ = read_video_av(
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
        del img
    del vframes
    if return_msg:
        exit('return_msg not implemented yet')
    else:
        return images_group


def collate_fn(batch):
    return batch


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        required=False,
        default="eval_mvbench",
    )
    parser.add_argument(
        "--pooling_shape",
        type=str,
        required=False,
        default="16-20-20",
    )
    parser.add_argument(
        "--error_message",
        type=str,
        required=False,
        default='error occured during captioning',
    )
    args = parser.parse_args()
    return args


def load_model_and_dataset(
    rank,
    world_size,
    pretrained_model_name_or_path,
    num_frames,
    use_lora,
    lora_alpha,
    weight_dir,
    csv_path,
    pooling_shape=(16, 12, 12)):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(
        pretrained_model_name_or_path,
        num_frames=num_frames,
        use_lora=use_lora,
        weight_dir=weight_dir,
        lora_alpha=lora_alpha,
        pooling_shape=pooling_shape,
    )
    model = model.half()
    logger.info("done loading llava")

    #  position embedding
    # model = model.to(torch.device(rank))
    model = model.to(torch.device(f"cuda:{rank}"))  # 明确指定 GPU
    model = model.eval()

    dataset = CSVDataset(csv_path, num_frames)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset


def infer(
    model,
    processor,
    video_list,
    conv_mode,
    print_res=False):
    # check if any video in video_list is None, if so, raise an exception
    if any([video is None for video in video_list]):
        raise Exception("Video not loaded properly")
    conv = conv_template.copy()
    conv.user_query("Describe the video in details.", is_mm=True)

    llm_responses, conv = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        video_list=video_list,
        max_new_tokens=500, # 256,   500
        do_sample=False,
        print_res=print_res,
    )

    return llm_responses


def run(rank, args, world_size, output_queue):
    # 设置进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)  # 每个进程只能看到自己的 GPU，这里固定为 0
    if rank == 0:
        if os.getenv("DEBUG_ADDRESS") != None:
            import ptvsd

            ptvsd.enable_attach(address=("localhost", int(os.getenv("DEBUG_ADDRESS"))), redirect_output=True)
            ptvsd.wait_for_attach()
            print("waiting for debugger attachment")
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = False
    conv_mode = args.conv_mode
    if args.pooling_shape is not None:
        pooling_shape = tuple([int(x) for x in args.pooling_shape.split("-")])

    logger.info(f"loading model and constructing dataset to gpu {rank}...")
    model, processor, dataset = load_model_and_dataset(
        rank,
        world_size,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
        pooling_shape=pooling_shape,
        csv_path=args.csv_path,
    )
    logger.info(f"done model and dataset...")
    logger.info("constructing dataset...")
    logger.info("single test...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    total = 0
    result_list = []
    for batch in tqdm(dataloader):
        total += 1
        try:
            preds = infer(
                model,
                processor,
                batch,
                conv_mode=conv_mode,
                print_res=print_res,
            )
        except Exception as e:
            logger.error(f"error in {batch}: {str(e)}")
            # preds = args.error_message duplicated for each video in the batch
            preds = [args.error_message] * len(batch)
        result_list.extend(preds)
    output_queue.put((rank, result_list))
    return result_list


def main():
    multiprocess = True
    mp.set_start_method("spawn")
    args = parse_args()
    if multiprocess:
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print(f"world_size: {world_size}")
        # Create a queue to collect results from each process
        output_queue = Queue()

        processes = []
        for i in range(world_size):
            # Each process will now also take the output queue as an argument
            p = Process(target=run, args=(i, args, world_size, output_queue))
            p.daemon = False
            processes.append(p)
            p.start()

        results_by_rank = {}
        for _ in range(world_size):
            rank, results = output_queue.get()  # Retrieve results as they finish
            results_by_rank[rank] = results
            print(f"Results received from rank {rank}")
            # ORDER THE RESULTS BY RANK
        logger.info("finished running")
        for p in processes:
            p.join()

        results_list = list(itertools.chain.from_iterable(results_by_rank[i] for i in range(world_size)))
    else:
        results_list = run(0, world_size=1, args=args)  # debug

    df = pd.read_csv(args.csv_path)
    # add a new column to the dataframe
    df["text"] = results_list
    drop_failed = True
    if drop_failed:
        # iterate through the dataframe and delete the entire row if captioning failed
        for i in tqdm(range(len(df))):
            if df["text"][i] == args.error_message:
                df = df.drop(i)
    # write the dataframe to a new csv file called '*_pllava_13b_caption.csv'
    new_csv_path = args.csv_path.replace(".csv", "_text.csv")
    df.to_csv(new_csv_path, index=False)
    print(f"Results saved to {new_csv_path}")


if __name__ == "__main__":
    main()


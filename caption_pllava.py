import sys
import os
import os
from pathlib import Path

current_file = Path(__file__)  # Gets the path of the current file
fourth_level_parent = current_file.parents[3]

datasets_dir = os.path.join(fourth_level_parent, "opensora/datasets")
import sys
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
    # system="Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    # system="Describe this video. Pay attention to all the people, objects, and their interactions in the video. Please describe the content of the video and the changes that occur, in chronological order. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A curious little boy, wearing a bright red T-shirt, stands in a sunny kitchen. The setting shows a kitchen scene, complete with a refrigerator and table. The boy's small hands hold the stainless steel handle of a modern refrigerator tightly, and he is slowly opening the door. The lights inside the refrigerator illuminate the colorful variety of fruits, vegetables and snacks, casting a soft glow on his excited face. He leans over and stares at the shelves with a look of surprise and joy. The overall environment is warm and harmonious, capturing the simple joys of a child's daily life. 2. An elderly gentleman with a calm face sits by the water, dressed in casual clothes, with a steaming cup of tea at his side. He has a paintbrush in his hand, carefully painting an oil painting on a small canvas. The sea breeze blows through his silver hair, gently blowing his loose white shirt, while the salty air adds an intangible element to his ongoing masterpiece, which is captured on the artist's canvas. The setting sun reflects on the calm sea, presenting vibrant colors. 3. In a warmly lit kitchen, a pair of skilled hands, adorned with a simple silver band on the right ring finger, gently grasp a polished stainless steel knife. The fingers, with neatly trimmed nails, exhibit a blend of strength and precision as they begin the delicate task of slicing through a freshly baked, golden-brown loaf of bread. The knife's blade glides effortlessly, revealing the soft, airy interior of the bread, which contrasts beautifully with its crisp crust. The close-up shots capture the deformation of the bread as it is manipulated and the subtle changes in light on the blade, creating a scene that is both serene and filled with anticipation for a home-cooked meal.",
    # system="Please describe the content of this video in as much detail as possible, including the people, objects, scenery, environment, and camera movements within the video and focus on describing the interaction between people and objects in the video. Please describe the content of the video and the changes that occur, in chronological order. The description should be useful for AI to re-generate the video. The description should not be less than six sentences. Here are some examples of good descriptions: 1. A young boy dressed in a vibrant red T-shirt standing in a well-lit kitchen bathed in sunlight. The kitchen is a typical domestic setting, featuring a refrigerator with a stainless steel handle and a table nearby. The boy's small hands grip the refrigerator handle as then he slowly pulls the door open. Inside, the refrigerator's lights reveal a colorful array of fruits, vegetables, and snacks, creating a soft, inviting glow that highlights the boy's beaming face. He leans in for a closer look, his eyes wide with surprise and delight at the sight of the food. The video is a heartwarming depiction of a child's innocent exploration of a simple household appliance, capturing the joy and amazement that can be found in the everyday moments of life. 2. In a warmly lit kitchen, a pair of skilled hands, adorned with a simple silver band on the right ring finger, gently grasp a polished stainless steel knife. The fingers, with neatly trimmed nails, exhibit a blend of strength and precision as they begin the delicate task of slicing through a freshly baked, golden-brown loaf of bread. The knife's blade glides effortlessly, revealing the soft, airy interior of the bread, which contrasts beautifully with its crisp crust. The close-up shots capture the deformation of the bread as it is manipulated and the subtle changes in light on the blade, creating a scene that is both serene and filled with anticipation for a home-cooked meal.",
    # system="Please describe the content of this video in as much detail as possible, including the people, objects, scenery, environment, and camera movements within the video and focus on describing the interaction between people and objects in the video. Please describe the content of the video and the changes that occur, in chronological order. The description should be useful for AI to re-generate the video. The description should not be less than six sentences.",
    #system="Please describe the content of this video in as much detail as possible, including the people, objects, scenery, environment, and camera movements within the video. Focus on describing the interaction between people and objects in the video, such as what actions people take, changes in the position or shape of objects, etc. Please describe the content of the video and the changes that occur, in chronological order. The description should be useful for AI to re-generate the video. The description should not be less than six sentences.",
    system="Please describe the content of this video in as much detail as possible. Please describe the content of the video and the changes that occur, in chronological order. \
            The description should be useful for AI to re-generate the video. The description should not be less than six sentences. Here is one example of good descriptions: \
                1. There is only one person in the video, a young girl of yellow race. She is wearing a black sling and has brown curly hair that falls on her shoulders and back. \
                She has a fair figure, big eyes, delicate skin, deep and bright eyes, and blood-red lips. She looks sexy and has delicate makeup. \
                Her expression is calm at first, but then she smiles seductively at the camera. \
                The background is the blurred outline of city buildings and windows, through which you can see the scenery outside.",
    # system="Describe this video. Pay attention to all the people, objects, and the interactions between the human and object in the video. The description should be useful for AI to re-generate the video. The description should be no more than 12 sentences. Here are some examples of good descriptions: \n1. A curious little boy, wearing a bright red T-shirt that stands out against the sunlit kitchen, is filled with a sense of wonder. His eyes are wide with excitement, and a slight smile plays on his lips as he anticipates the treasures within the refrigerator. Standing on tiptoe, he leans slightly forward, his small hands gripping the cool, stainless steel handle with determination. His breath quickens in eager anticipation, capturing the innocence and curiosity of childhood. The refrigerator is a sleek, modern appliance, its stainless steel surface gleaming in the sunlight that filters through the kitchen window. Inside, an array of colorful fruits, vegetables, and snacks are neatly arranged. Shiny red apples, bunches of golden bananas, and crisp green lettuce create a vibrant display. A variety of yogurt cups in bright packaging and a jar of homemade strawberry jam add to the enticing collection, each item perfectly positioned as if inviting exploration and enjoyment. As the boy slowly pulls open the refrigerator door, the gentle hum of the appliance fills the room. His fingers trace the edge of the door, feeling the smooth, cold metal beneath his touch. His eyes dart from shelf to shelf, captivated by the colors and possibilities. Tentatively, he reaches out and touches an apple, feeling its smooth, cool surface. He quickly withdraws his hand, savoring the thrill of this small adventure, his face lit with pure delight. The kitchen exudes warmth and harmony, with sunlight streaming through the window, casting playful patterns on the tiled floor. A polished wooden table stands nearby, adorned with a vase of fresh daisies that add a splash of color and life to the room. Everything is serene and inviting, creating a perfect backdrop for this moment of simple joy. The scene captures the essence of a child's daily life, where even the most ordinary moments are filled with wonder and happiness. \nAn elderly gentleman with a serene expression sits by the water. His silver hair dances softly in the sea breeze, framing a visage that exudes calm and focus. Dressed in comfortable, casual attire, he wears a loose white shirt that flutters gently with each gust of wind. His eyes, sharp yet gentle, are fixed intently on the canvas before him, reflecting a lifetime of experiences and a deep connection with his art. Beside him, a steaming cup of tea sits on a weathered wooden stool, its aromatic tendrils mingling with the salty sea air. In his hand, he holds a paintbrush, its bristles stained with vibrant hues of oil paint. The small canvas perched on an easel captures the essence of the setting sun and the calm sea, each stroke of color meticulously applied to bring the scene to life. The paints, in various shades of blues, oranges, and purples, are laid out in a well-used palette, each color carefully mixed to achieve the perfect tone. With deliberate, graceful movements, the gentleman dips his brush into the palette, selecting a deep blue to capture the sea's tranquility. His hand moves with practiced precision, each stroke deliberate and infused with emotion. As he paints, the brush glides over the canvas, leaving trails of color that mimic the vibrant reflections of the setting sun on the water. Occasionally, he pauses to sip his tea, savoring the warmth and flavor before returning to his work, the cup leaving a faint ring of moisture on the stool. The overall scene is one of peaceful solitude, where nature and creativity blend seamlessly. The setting sun casts a golden glow over the calm sea, painting the horizon with vibrant colors that mirror the artist's canvas. The gentle lapping of waves provides a soothing soundtrack, while seagulls call in the distance, adding to the serene ambiance. This tranquil setting, with its harmonious blend of colors and sounds, encapsulates a moment of pure artistry and reflection, where time seems to stand still, allowing the artist to capture the beauty of the world around him. \nIn a warmly lit kitchen, a pair of skilled hands takes center stage, their movements confident and assured. The hands belong to someone with a keen sense of culinary artistry, evidenced by the simple silver band on the right ring finger, hinting at a life filled with cherished moments. The fingers, with neatly trimmed nails, reflect both strength and precision, suggesting years of experience in the kitchen. The gentle yet firm grip on the knife speaks volumes about the person’s dedication to crafting the perfect meal. The polished stainless steel knife gleams under the kitchen lights, its sharp blade a testament to its quality and utility. It is a tool well-cared for, with a handle that fits perfectly into the hand, balancing elegance and functionality. The freshly baked loaf of bread, golden-brown and aromatic, rests on a wooden cutting board. Its crust is crisp, offering a satisfying resistance to the knife, while the interior is soft and airy, releasing a warm, inviting aroma that fills the room. With a steady hand, the knife is guided through the loaf, each slice executed with precision. The blade glides effortlessly, a seamless dance between steel and bread, revealing the soft interior with each cut. As the knife moves, the bread deforms slightly under the pressure, the crust crackling softly. The person’s fingers adjust subtly, maintaining control and ensuring uniform slices. Light catches the blade, creating a play of reflections that highlight the meticulous nature of the task, each motion deliberate and filled with anticipation. The kitchen is a haven of warmth and comfort, illuminated by soft, ambient lighting that casts gentle shadows across the countertops. The aroma of freshly baked bread mingles with the faint scent of herbs and spices, creating an inviting atmosphere. Nearby, a rustic wooden table is set for a meal, complete with simple, elegant place settings. The scene is serene, a perfect blend of craftsmanship and homeliness, promising the simple joy of a home-cooked meal shared with loved ones.",
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token="<image>",
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672  #


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
    stop_criteria_keywords=None,
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


# def load_video(video_path, num_frames, return_msg=False, resolution=336):
#     transforms = torchvision.transforms.Resize(size=resolution)
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     total_num_frames = len(vr)
#     frame_indices = get_index(total_num_frames, num_frames)
#     images_group = list()
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy())
#         images_group.append(transforms(img))
#     if return_msg:
#         fps = float(vr.get_avg_fps())
#         sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
#         # " " should be added in the start and end
#         msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
#         return images_group, msg
#     else:
#         return images_group


def load_video(video_path, num_frames, return_msg=False, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vframes, aframes, info = read_video_av(
        video_path,
        pts_unit="sec", 
        output_format="THWC"
    )
    print(vframes.shape)
    total_num_frames = len(vframes)
    # print("Video path: ", video_path)
    # print("Total number of frames: ", total_num_frames)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vframes[frame_index].numpy())
        images_group.append(transforms(img))
    if return_msg:
        # fps = float(vframes.get_avg_fps())
        # sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # # " " should be added in the start and end
        # msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        # return images_group, msg
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
        default=None,
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
    pooling_shape=(16, 12, 12),
):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(
        pretrained_model_name_or_path,
        num_frames=num_frames,
        use_lora=use_lora,
        weight_dir=weight_dir,
        lora_alpha=lora_alpha,
        pooling_shape=pooling_shape,
    )
    logger.info("done loading llava")

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = CSVDataset(csv_path, num_frames)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset


def infer(
    model,
    processor,
    video_list,
    conv_mode,
    print_res=False,
):
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
        max_new_tokens=300, # 256,   500
        do_sample=False,
        print_res=print_res,
    )

    return llm_responses


def run(rank, args, world_size, output_queue):
    if rank == 0:
        import os

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
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    total = 0
    result_list = []
    print(len(dataset))
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
    # csv_path = '/home/tom/PLLaVA/test_short_caption_part2.csv'
    if multiprocess:
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print(f"world_size: {world_size}")
        # Create a queue to collect results from each process
        output_queue = Queue()

        # with Pool(world_size) as pool:
        #     func = functools.partial(run, args=args, world_size=world_size)
        #     result_lists = pool.map(func, range(world_size))
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
        # results_list = list(itertools.chain([results_by_rank[i] for i in range(world_size)]))
        # (data[key] for key in sorted_keys)
        # results_list = [item for sublist in results_by_rank.values() for item in sublist]

    else:
        results_list = run(0, world_size=1, args=args)  # debug

    print(results_list)

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


import argparse
import os
import gc
import queue
import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision.transforms.functional import pil_to_tensor
from tools.scoring.optical_flow.unimatch import UniMatch
import cv2

def extract_frames(video_path, frame_inds):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in frame_inds:
        idx = min(idx, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    return frames

def process_video(idx, video_path, frame_inds, model, device):
    frames = extract_frames(video_path, frame_inds)
    if len(frames) < len(frame_inds):
        return idx, -1.0  # error or short video

    images = torch.stack([pil_to_tensor(f).float() for f in frames])  # [N, C, H, W]
    images = images.to(f"cuda:{device}")
    H, W = images.shape[-2:]
    if H > W:
        images = rearrange(images, "N C H W -> N C W H")
    images = torch.nn.functional.interpolate(images, size=(320, 576), mode="bilinear", align_corners=True)

    with torch.no_grad():
        b0 = rearrange(images[:-1], "N C H W -> (N) C H W").contiguous()
        b1 = rearrange(images[1:], "N C H W -> (N) C H W").contiguous()

        res = model(
            b0,
            b1,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task="flow",
            pred_bidir_flow=False,
        )
        flow_map = res["flow_preds"][-1]  # (N-1, 2, H, W)
        flow_map = rearrange(flow_map, "N C H W -> N H W C")
        score = flow_map.norm(dim=-1).mean().item()

    return idx, score

import time

def gpu_worker(gpu_id, model_path, task_queue, result_queue, frame_inds, max_retries=5, retry_delay=5):
    """ GPU绑定的长期工作进程 """
    torch.cuda.set_device(gpu_id)
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    ).to(f"cuda:{gpu_id}")
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()

    retry_counts = {}  # 记录每个任务的重试次数

    while True:
        task = task_queue.get()
        if task is None:
            break  # 终止信号

        idx, video_path = task

        # 每个任务最多重试 max_retries 次
        retries = retry_counts.get(idx, 0)
        
        while retries < max_retries:
            try:
                result = process_video(idx, video_path, frame_inds, model, gpu_id)
                result_queue.put(result)
                retry_counts[idx] = 0  # Reset retry count on success
                break  # 任务成功，退出重试循环
            except torch.cuda.OutOfMemoryError as e:
                print(f"[GPU {gpu_id}] CUDA OOM on task {task}, retrying... (attempt {retries+1}/{max_retries})")
                retries += 1
                retry_counts[idx] = retries
                if retries < max_retries:
                    time.sleep(retry_delay)  # 等待一段时间，允许显存释放
                else:
                    print(f"[GPU {gpu_id}] Task {task} failed after {max_retries} attempts.")
                    result_queue.put((idx, -1.0))  # 将任务失败标记为 -1
            except Exception as e:
                print(f"[GPU {gpu_id}] Error on task {task}: {e}")
                result_queue.put((idx, -1.0))  # 将其他错误标记为失败
                break  # 对于其他错误，退出任务循环
            finally:
                torch.cuda.empty_cache()  # 释放显存



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--model_path", type=str, default="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/weights/unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth", required=False)
    parser.add_argument("--pool_per_gpu", type=int, default=2, help="Number of processes per GPU")
    return parser.parse_args()

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    df = pd.read_csv(args.meta_path)
    video_paths = df["path"].tolist()

    frame_inds = [15 * i for i in range(10)]
    num_gpus = torch.cuda.device_count()
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 启动子进程
    for gpu_id in range(num_gpus):
        for _ in range(args.pool_per_gpu):
            mp.Process(
                target=gpu_worker,
                args=(gpu_id, args.model_path, task_queue, result_queue, frame_inds)
            ).start()

    # 分配任务
    for idx, path in enumerate(video_paths):
        task_queue.put((idx, path))

    # 添加终止信号
    for _ in range(num_gpus * args.pool_per_gpu):
        task_queue.put(None)

    pbar = tqdm(total=len(video_paths), desc="Processing")
    completed = 0
    failed = 0
    while completed < len(video_paths):
        try:
            idx, flow_score = result_queue.get(timeout=300)  # 最多等待5分钟
            if flow_score == -1.0:
                failed += 1  # 记录失败任务
            else:
                df.at[idx, "flow"] = flow_score
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"GPU Utilization: {torch.cuda.utilization()}%")
        except queue.Empty:
            print("⚠️ 主进程等待结果超时，可能有子进程崩溃或卡死")
            break

    print(f"任务完成，成功任务: {completed}, 失败任务: {failed}")

    # ====== 保存结果 ======
    output_path = os.path.splitext(args.meta_path)[0] + "_flow.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 结果保存至: {output_path}")


if __name__ == "__main__":
    main()

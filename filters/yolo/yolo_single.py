import os
import torch
import pandas as pd
import sys
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import multiprocessing as mp  # 导入多进程模块

# 配置参数
BATCH_SIZE = 32
CONF_THRESH = 0.6
IOU_THRESH = 0.45
SAVE_INTERVAL = 10000

# 用于处理每个视频的函数
def process_video(idx, video_path, model, device):
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} load error, please check the video path.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = max(1, total_frames // 30)
    
    frame_buffer = []
    max_persons = 0
    overlap_flag = 0
    detected = False

    with torch.cuda.device(device):
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 动态帧采样策略
                if (cap.get(cv2.CAP_PROP_POS_FRAMES) % step_size) == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(frame)

                    # 批量推理（达到批次大小或视频结束）
                    if len(frame_buffer) == BATCH_SIZE or not ret:
                        with torch.inference_mode():
                            results = model(frame_buffer, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

                        for res in results:
                            boxes = res.boxes.xyxy.float()
                            cls_ids = res.boxes.cls
                            
                            # 筛选人体检测
                            person_mask = (cls_ids == 0)
                            person_boxes = boxes[person_mask]
                            person_count = person_mask.sum().item()

                            if person_count > 0:
                                detected = True
                                max_persons = max(max_persons, person_count)

                        frame_buffer.clear()
                        
                        del results
                        torch.cuda.empty_cache()

    cap.release()
    return idx, int(detected), max_persons, overlap_flag

# 处理单个CSV文件的函数
def process_single_csv(csv_path, gpu_id, output_csv, progress_counter):
    try:
        df = pd.read_csv(csv_path)
        print(f"\n开始处理文件: {csv_path}")
    except Exception as e:
        print(f"文件读取失败: {csv_path} | 错误: {str(e)}")
        return

    # 初始化结果列
    for col in ["has_person", "num_persons", "bbox_overlap"]:
        if col not in df.columns:
            df[col] = 0

    video_paths = df["path"].tolist()
    if len(video_paths) == 0:
        print(f"文件无有效视频路径: {csv_path}")
        return

    # 初始化模型
    model = YOLO("/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/weights/yolo/yolo11m.pt ", verbose=False)
    model.to(f"cuda:{gpu_id}")
    model.model.eval()

    # 使用 tqdm 包装循环以显示进度条
    with tqdm(total=len(video_paths), desc=f"Processing {os.path.basename(csv_path)}") as pbar:
        # 使用多进程池处理每个视频
        with mp.Pool(processes=4) as pool:  # 创建4个子进程来处理
            results = []
            for idx, video_path in enumerate(video_paths):
                results.append(pool.apply_async(process_video, (idx, video_path, model, gpu_id)))

            # 等待所有任务完成并更新进度条
            for result in results:
                idx, has_person, num_persons, bbox_overlap = result.get()
                df.at[idx, "has_person"] = has_person
                df.at[idx, "num_persons"] = num_persons
                df.at[idx, "bbox_overlap"] = int(bbox_overlap)
                progress_counter.value += 1  # 更新进度计数
                pbar.update(1)  # 每处理完一个视频更新一次进度条

    # 保存处理结果到不同的 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"处理完成: {output_csv}")

# 主函数，接收参数
if __name__ == "__main__":
    # 设置多进程的启动方法为 'spawn'
    mp.set_start_method('spawn', force=True)
    
    if len(sys.argv) != 3:
        print("Usage: python process_csv.py <gpu_id> <csv_path>")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    csv_path = sys.argv[2]
    output_csv = os.path.join(os.path.dirname(csv_path), f"processed_{os.path.basename(csv_path)}")

    # 创建共享的进度计数器
    manager = mp.Manager()
    progress_counter = manager.Value('i', 0)

    process_single_csv(csv_path, gpu_id, output_csv, progress_counter)

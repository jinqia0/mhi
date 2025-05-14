import cv2
import os
import torch
import pandas as pd
import multiprocessing as mp
from ultralytics import YOLO
from tqdm import tqdm
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
torch.backends.cudnn.benchmark = True  # 启用cuDNN优化

# ====== 多GPU处理核心 ======
def calculate_iou(box1, box2):
    """计算两个框之间的IoU，支持N*N个框之间的计算"""
    
    # 计算交集坐标
    inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))  # 计算每对框的左上角x坐标
    inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))  # 计算每对框的左上角y坐标
    inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))  # 计算每对框的右下角x坐标
    inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))  # 计算每对框的右下角y坐标
    
    # 计算交集面积
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # 计算每个框的面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算IoU：交集面积 / 并集面积
    iou = inter_area / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area + 1e-6)
    
    return iou  # 返回形状为 (N, N) 的IoU矩阵

def process_video(idx, video_path, model, device, batch_size, conf_thresh, iou_thresh):
    """ 优化后的视频处理（支持批量推理，区分主要人物和统计其它相关指标） """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_size = max(1, total_frames // 30)
        
        frame_buffer = []
        max_persons = 0
        max_iou = 0
        max_main_persons = 0
        max_main_iou = 0
        detected = False

        with torch.cuda.device(device):  # 显式指定设备上下文
            with torch.no_grad():        # 禁用梯度计算
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # 动态帧采样策略
                    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % step_size) == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_buffer.append(frame)

                        # 批量推理（达到批次大小或视频结束）
                        if len(frame_buffer) == batch_size or not ret:
                            with torch.inference_mode():
                                results = model(frame_buffer, 
                                            conf=conf_thresh, 
                                            iou=iou_thresh, 
                                            verbose=False)

                            for res in results:
                                # 同步张量到CPU并转换类型
                                boxes = res.boxes.xyxy.float()
                                cls_ids = res.boxes.cls
                                
                                # 筛选人体检测
                                person_mask = (cls_ids == 0)
                                person_boxes = boxes[person_mask]
                                person_count = person_mask.sum().item()

                                # 关键修复：更新最大人数统计
                                if person_count > 0:
                                    max_persons = max(max_persons, person_count)

                                # 主要人物统计
                                if person_boxes.shape[0] > 0:
                                    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1])
                                    max_area = areas.max()
                                    
                                    # 主要人物定义：面积 >= 最大面积的一半
                                    main_mask = areas >= (max_area * 0.5)
                                    main_person_boxes = person_boxes[main_mask]
                                    main_person_count = main_mask.sum().item()

                                    # 更新最大主要人物数
                                    max_main_persons = max(max_main_persons, main_person_count)

                                    # 计算主要人物间IoU
                                    if main_person_boxes.shape[0] >= 2:
                                        iou_matrix = calculate_iou(main_person_boxes, main_person_boxes)
                                        # 排除自己和自己之间的IoU（对角线）
                                        iou_matrix = iou_matrix - torch.eye(iou_matrix.size(0), device=iou_matrix.device)
                                        max_main_iou = max(max_main_iou, iou_matrix.max().item())

                                # 批量计算IoU（使用矩阵运算）
                                if person_boxes.shape[0] >= 2:
                                    iou_matrix = calculate_iou(person_boxes, person_boxes)
                                    iou_matrix = iou_matrix - torch.eye(iou_matrix.size(0), device=iou_matrix.device)
                                    # 获取所有人物的最大 IoU
                                    max_iou = max(max_iou, iou_matrix.max().item())

                            frame_buffer.clear()

        cap.release()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return idx, -1, -1, -1, -1, -1

    return idx, max_persons, max_iou, max_main_persons, max_main_iou

def gpu_worker(gpu_id, task_queue, result_queue, args):
    """ GPU绑定的长期工作进程 """
    torch.cuda.set_device(gpu_id)
    model = YOLO("/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/weights/yolo/yolo11m.pt", verbose=False)  # 修复模型版本
    model.to(f"cuda:{gpu_id}")
    model.model.eval()  # 设置为评估模式
    
    # 优化显存配置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    while True:
        task = task_queue.get()
        if task is None: break  # 终止信号
        result = process_video(*task, model=model, device=gpu_id, iou_thresh=args.iou_thresh, conf_thresh=args.conf_thresh, batch_size=args.batch_size)
        result_queue.put(result)
        del result
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    # ====== 使用 argparse 解析命令行参数 ======
    parser = argparse.ArgumentParser(description="Process video data and update CSV")
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--pool_size_per_gpu', type=int, default=1, help='Number of processes per GPU')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--iou_thresh', type=float, default=0.4, help='IoU threshold for detection')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--chunksize', type=int, default=1000, help='Chunk size for processing CSV')
    parser.add_argument('--temp_dir', type=str, default='./temp', help='Temporary directory for saving chunks')
    parser.add_argument('--remain_temp', action='store_true', help='Do not clean temporary files after processing')
    args = parser.parse_args()

    # ====== 初始化并行环境 ======
    mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    
    # 读取CSV文件，分批处理
    chunksize = args.chunksize
    chunk_idx = 0
    os.makedirs(args.temp_dir, exist_ok=True)  # 创建临时目录
    file_name = os.path.basename(args.csv_path).split(".")[0]
    for chunk in pd.read_csv(args.csv_path, usecols=["path"], chunksize=chunksize):
        df = chunk
        video_paths = df["path"].tolist()
            
        # ====== 创建任务队列 ======
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # ====== 启动GPU工作进程 ======
        for gpu_id in range(num_gpus):
            for _ in range(args.pool_size_per_gpu):  # 每个GPU启动多个进程
                mp.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, args)).start()

        # ====== 填充任务队列 ======
        for idx, path in enumerate(video_paths):
            task_queue.put((idx, path))

        # ====== 添加终止信号 ======
        for _ in range(num_gpus * args.pool_size_per_gpu):
            task_queue.put(None)

        # ====== 结果收集 ====== 
        pbar = tqdm(total=len(video_paths), desc="Processing")
        completed = 0
        while completed < len(video_paths):
            idx, num_persons, max_iou, num_main_persons, max_main_iou = result_queue.get()
            df.at[idx, "num_persons"] = num_persons
            df.at[idx, "max_iou"] = float(max_iou)
            df.at[idx, "num_main_persons"] = num_main_persons
            df.at[idx, "max_main_iou"] = float(max_main_iou)
            completed += 1
            pbar.update(1)
            pbar.set_postfix_str(f"GPU Utilization: {torch.cuda.utilization()}%")

        pbar.close()
        # ====== 保存结果 ======
        chunk_idx += 1
        temp_path = os.path.join(args.temp_dir, f"{file_name}_chunk_{chunk_idx}.csv")
        df.to_csv(temp_path, index=False)
        print(f"第{chunk_idx + 1}批数据处理完成，保存到文件: {temp_path}")
    
    # ====== 合并结果 ======
    # 保存结果的CSV文件路径
    output_path = os.path.splitext(args.csv_path)[0] + "_yolo.csv"
    os.remove(output_path) if os.path.exists(output_path) else None  # 删除旧文件
    final_df = pd.concat([pd.read_csv(os.path.join(args.temp_dir, f"{file_name}_chunk_{i+1}.csv")) for i in range(chunk_idx)], ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"所有数据处理完成，结果保存到文件: {output_path}")
    # 清理临时文件
    if not args.remain_temp:
        for i in range(chunk_idx):
            os.remove(os.path.join(args.temp_dir, f"{file_name}_chunk_{i}.csv"))
            print(f"临时文件已删除")
    
#!/usr/bin/env python3
"""
Qwen2.5-VL-72B-Instruct Video Caption Inference Script

基于Hugging Face框架使用Qwen2.5-VL-72B-Instruct模型进行视频caption推理
"""

import os
import sys
import logging
import argparse
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
import itertools

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2

# 设置环境变量
os.environ.update({
    'PYTHONWARNINGS': 'ignore',
    'TOKENIZERS_PARALLELISM': 'false',
})

# PyTorch优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Caption提示词 - 修改这里来自定义提示词内容
CAPTION_PROMPT = """Please provide a structured and detailed description of this video following this EXACT format:

<scene>Describe the setting, environment, background, lighting, and overall atmosphere of the video in detail</scene>;<person>Describe each person visible in the video, including their appearance, clothing, age, gender, expressions, and body language</person>;<interaction>Describe the interactions between people, their movements, actions, and how they relate to each other throughout the video</interaction>

You MUST use exactly this format with the XML-style tags and semicolons as separators. Do not add any other text outside of this structure."""

def format_structured_caption(caption_text):
    """
    解析和格式化XML标签格式的结构化caption输出
    
    Args:
        caption_text: 原始caption文本（应包含XML标签格式）
        
    Returns:
        formatted_caption: 格式化后的结构化caption
    """
    import re
    
    try:
        # 移除多余的空白和换行
        caption_text = ' '.join(caption_text.split())
        
        # 使用正则表达式提取XML标签内容
        scene_match = re.search(r'<scene>(.*?)</scene>', caption_text, re.IGNORECASE | re.DOTALL)
        person_match = re.search(r'<person>(.*?)</person>', caption_text, re.IGNORECASE | re.DOTALL)
        interaction_match = re.search(r'<interaction>(.*?)</interaction>', caption_text, re.IGNORECASE | re.DOTALL)
        
        formatted_parts = []
        
        # 提取场景描述
        if scene_match:
            scene_text = scene_match.group(1).strip()
            if scene_text:
                formatted_parts.append(f"<scene>{scene_text}</scene>")
        
        # 提取人物描述
        if person_match:
            person_text = person_match.group(1).strip()
            if person_text:
                formatted_parts.append(f"<person>{person_text}</person>")
        
        # 提取交互描述
        if interaction_match:
            interaction_text = interaction_match.group(1).strip()
            if interaction_text:
                formatted_parts.append(f"<interaction>{interaction_text}</interaction>")
        
        # 如果成功提取到标签格式的内容，返回格式化结果
        if formatted_parts:
            return ';'.join(formatted_parts)
        
        # 如果没有找到XML标签，检查是否包含分号分隔的格式
        if ';' in caption_text:
            parts = caption_text.split(';')
            if len(parts) == 3:
                # 尝试为没有标签的内容添加标签
                scene_text = parts[0].strip()
                person_text = parts[1].strip()
                interaction_text = parts[2].strip()
                
                formatted_parts = [
                    f"<scene>{scene_text}</scene>",
                    f"<person>{person_text}</person>",
                    f"<interaction>{interaction_text}</interaction>"
                ]
                return ';'.join(formatted_parts)
        
        # 如果都没有匹配，返回原文本，但添加提示
        logger.warning("Failed to parse structured format, returning original text")
        return f"<general>{caption_text}</general>"
        
    except Exception as e:
        logger.warning(f"Error formatting structured caption: {e}")
        return f"<error>{caption_text}</error>"

def extract_frames_from_video(video_path, num_frames=8, max_size=512):
    """
    从视频中提取指定数量的帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        max_size: 图像最大尺寸
    
    Returns:
        frames: PIL Image列表
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # BGR转RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像并调整大小
            pil_image = Image.fromarray(frame_rgb)
            
            # 保持宽高比的调整大小
            width, height = pil_image.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            frames.append(pil_image)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
            
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        raise e

def load_qwen_model(model_id, device_id=0, use_distributed=True):
    """
    加载Qwen2.5-VL模型和处理器，支持分布式加载
    
    Args:
        model_id: Hugging Face模型ID
        device_id: 主GPU设备ID (仅在非分布式模式下使用)
        use_distributed: 是否使用分布式加载
    
    Returns:
        model, processor
    """
    try:
        logger.info(f"Loading model from Hugging Face: {model_id}")
        
        # 检查可用GPU数量
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        
        if use_distributed and num_gpus > 1:
            logger.info("Using distributed loading across multiple GPUs")
            # 自动分布式加载到所有可用GPU
            device_map = "auto"
        else:
            logger.info(f"Using single GPU: cuda:{device_id}")
            device_map = f"cuda:{device_id}"
        
        # 加载模型 - 支持分布式
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # 直接使用float16
                device_map=device_map,
                max_memory={i: "16GiB" for i in range(num_gpus)} if use_distributed else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="./temp_offload" if use_distributed else None,
            )
            logger.info("Successfully loaded with auto dtype and device_map")
        except Exception as e:
            logger.warning(f"Failed to load with auto settings: {e}")
            logger.info("Trying with float16 and manual memory management...")
            try:
                # 手动设置每个GPU的最大内存使用量
                if use_distributed and num_gpus > 1:
                    max_memory_dict = {}
                    for i in range(num_gpus):
                        # 为每个GPU分配更保守的内存限制
                        max_memory_dict[i] = "14GiB"  # 更保守的内存分配
                else:
                    max_memory_dict = {device_id: "20GiB"}
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory_dict,
                    low_cpu_mem_usage=True,
                    offload_folder="/tmp/qwen_offload" if use_distributed else None
                )
                logger.info("Successfully loaded with float16 and memory limits")
            except Exception as e2:
                logger.error(f"Distributed loading failed: {e2}")
                logger.info("Trying CPU loading with gradual GPU transfer...")
                try:
                    # 最后的尝试：CPU加载然后移动到GPU
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                    
                    # 如果是单GPU模式，移动到指定GPU
                    if not use_distributed or num_gpus == 1:
                        logger.info(f"Moving model to cuda:{device_id}")
                        model = model.to(f"cuda:{device_id}")
                    else:
                        # 多GPU环境下，手动分配层到不同GPU
                        logger.info("Manually distributing model layers across GPUs")
                        model = distribute_model_layers(model, num_gpus)
                        
                except Exception as e3:
                    logger.error(f"All loading attempts failed: {e3}")
                    raise e3
                    
        model.eval()
        
        # 加载处理器
        processor = AutoProcessor.from_pretrained(model_id)
        
        logger.info("Successfully loaded Qwen2.5-VL model and processor")
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model from {model_id}: {e}")
        raise e

def distribute_model_layers(model, num_gpus):
    """
    手动将模型层分布到多个GPU上
    
    Args:
        model: 已加载的模型
        num_gpus: GPU数量
    
    Returns:
        分布式模型
    """
    logger.info(f"Distributing model layers across {num_gpus} GPUs")
    
    # 获取模型的所有层
    total_layers = len(model.model.layers) if hasattr(model.model, 'layers') else 32
    layers_per_gpu = total_layers // num_gpus
    
    try:
        # 将embedding和其他组件放在第一个GPU
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(f"cuda:0")
        
        # 分配transformer层到不同GPU
        for i, layer in enumerate(model.model.layers):
            gpu_id = min(i // layers_per_gpu, num_gpus - 1)
            layer = layer.to(f"cuda:{gpu_id}")
            logger.debug(f"Layer {i} -> GPU {gpu_id}")
        
        # 将最后的layer norm和lm_head放在最后一个GPU
        if hasattr(model.model, 'norm'):
            model.model.norm = model.model.norm.to(f"cuda:{num_gpus-1}")
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(f"cuda:{num_gpus-1}")
            
        logger.info("Successfully distributed model layers")
        return model
        
    except Exception as e:
        logger.error(f"Failed to distribute model layers: {e}")
        # 如果手动分布失败，尝试简单的to()方法
        return model.to(f"cuda:0")

def generate_caption(model, processor, frames, device):
    """
    为视频帧生成描述
    
    Args:
        model: Qwen2.5-VL模型
        processor: 处理器
        frames: 视频帧列表
        device: 设备
    
    Returns:
        caption: 生成的描述文本
    """
    try:
        # 构造消息
        messages = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video",
                        "video": frames,  # 直接传入PIL Image列表
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": CAPTION_PROMPT
                    }
                ]
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # 生成配置 - 移除不支持的参数
        generation_config = {
            "max_new_tokens": 256,  # 减少生成长度避免内存问题
            "do_sample": False,  # 使用贪婪解码获得确定性结果
            "repetition_penalty": 1.05,
        }
        
        # 生成文本 - 添加超时和内存管理
        try:
            with torch.no_grad():
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                logger.info("Starting caption generation...")
                generated_ids = model.generate(**inputs, **generation_config)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 清理临时变量
                del generated_ids, generated_ids_trimmed, inputs
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"CUDA OOM during generation: {e}")
                torch.cuda.empty_cache()
                # 尝试降低batch size再次生成
                return "Error: GPU memory insufficient for caption generation"
            else:
                raise e
        
        # 清理输出文本
        output_text = output_text.strip()
        
        # 格式化结构化输出
        output_text = format_structured_caption(output_text)
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return f"Error: {str(e)}"

def process_single_video(video_path, model, processor, device, num_frames=8):
    """
    处理单个视频文件
    
    Args:
        video_path: 视频路径
        model: 模型
        processor: 处理器
        device: 设备
        num_frames: 提取帧数
    
    Returns:
        caption: 生成的描述
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return f"Error: File not found - {video_path}"
        
        # 提取视频帧
        frames = extract_frames_from_video(video_path, num_frames=num_frames)
        
        # 生成描述
        caption = generate_caption(model, processor, frames, device)
        
        return caption
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return f"Error: {str(e)}"

def worker_process(rank, world_size, model_id, csv_path, output_queue, args):
    """
    工作进程函数
    
    Args:
        rank: 进程编号
        world_size: 总进程数
        model_id: Hugging Face模型ID
        csv_path: CSV文件路径
        output_queue: 输出队列
        args: 参数
    """
    try:
        # 设置GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        device = f"cuda:0"  # 每个进程只看到一个GPU
        
        # 加载模型
        logger.info(f"Process {rank}: Loading model...")
        model, processor = load_qwen_model(model_id, device_id=0)
        
        # 加载数据
        df = pd.read_csv(csv_path)
        video_paths = df['path'].tolist()
        
        # 分配数据给当前进程
        data_per_process = len(video_paths) // world_size
        start_idx = rank * data_per_process
        end_idx = (rank + 1) * data_per_process if rank < world_size - 1 else len(video_paths)
        
        process_video_paths = video_paths[start_idx:end_idx]
        
        logger.info(f"Process {rank}: Processing {len(process_video_paths)} videos ({start_idx}-{end_idx})")
        
        # 处理视频
        results = []
        for video_path in tqdm(process_video_paths, desc=f"GPU-{rank}"):
            caption = process_single_video(
                video_path, model, processor, device, 
                num_frames=args.num_frames
            )
            results.append(caption)
            
            # 定期清理GPU内存
            if len(results) % 10 == 0:
                torch.cuda.empty_cache()
        
        # 返回结果
        output_queue.put((rank, results))
        logger.info(f"Process {rank}: Completed processing")
        
    except Exception as e:
        logger.error(f"Process {rank}: Error - {e}")
        output_queue.put((rank, []))

def test_single_video_file(video_path, model_id, num_frames=8, use_distributed=True):
    """测试单个指定路径的视频文件"""
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    logger.info(f"Testing single video: {video_path}")
    logger.info(f"Using model: {model_id}")
    
    try:
        # 加载模型
        logger.info("Loading Qwen2.5-VL model...")
        model, processor = load_qwen_model(model_id, device_id=0, use_distributed=use_distributed)
        
        # 处理视频
        logger.info("Processing video...")
        caption = process_single_video(
            video_path, model, processor, "cuda:0", 
            num_frames=num_frames
        )
        
        logger.info("="*80)
        logger.info("RESULT:")
        logger.info("="*80)
        logger.info(f"Video: {video_path}")
        logger.info(f"Caption: {caption}")
        logger.info("="*80)
        
        # 保存结果到CSV文件
        output_path = os.path.join(os.path.dirname(video_path), 
                                  f"single_video_caption_{os.path.basename(video_path)}.csv")
        
        result_df = pd.DataFrame({
            'path': [video_path],
            'caption': [caption]
        })
        
        result_df.to_csv(output_path, index=False)
        logger.info(f"Result saved to: {output_path}")
        
        return caption
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Video Caption Inference")
    parser.add_argument("--model_id", type=str, 
                       default="Qwen/Qwen2.5-VL-72B-Instruct",
                       help="Hugging Face model ID for Qwen2.5-VL model")
    
    # 添加单视频测试选项
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to single video file for testing")
    
    parser.add_argument("--csv_path", type=str, default=None,
                       help="Path to CSV file containing video paths")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output CSV path (default: input_path + '_qwen_captions.csv')")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to extract from each video")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of worker processes (default: number of GPUs)")
    parser.add_argument("--single_gpu", action="store_true",
                       help="Use single GPU mode")
    parser.add_argument("--no_distributed", action="store_true",
                       help="Disable distributed model loading (force single GPU)")
    
    args = parser.parse_args()
    
    # 单视频测试模式
    if args.video_path:
        logger.info("Running in single video test mode")
        use_distributed = not args.no_distributed
        test_single_video_file(args.video_path, args.model_id, args.num_frames, use_distributed)
        return
    
    # 批量处理模式 - 需要CSV文件
    if not args.csv_path:
        logger.error("Either --video_path or --csv_path must be provided")
        logger.info("Usage examples:")
        logger.info("  Single video: python caption_qwen_vl.py --video_path /path/to/video.mp4")
        logger.info("  Batch mode:   python caption_qwen_vl.py --csv_path /path/to/videos.csv")
        logger.info("  Custom model: python caption_qwen_vl.py --video_path /path/to/video.mp4 --model_id Qwen/Qwen2.5-VL-7B-Instruct")
        return
    
    # 设置输出路径
    if args.output_path is None:
        args.output_path = args.csv_path.replace('.csv', '_qwen_captions.csv')
    
    # 检查输入文件
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV file not found: {args.csv_path}")
        return
    
    # 设置工作进程数
    if args.single_gpu or args.num_workers == 1:
        # 单GPU模式
        logger.info("Running in single GPU mode")
        
        model, processor = load_qwen_model(args.model_id, device_id=0)
        df = pd.read_csv(args.csv_path)
        video_paths = df['path'].tolist()
        
        results = []
        for video_path in tqdm(video_paths, desc="Processing videos"):
            caption = process_single_video(
                video_path, model, processor, "cuda:0", 
                num_frames=args.num_frames
            )
            results.append(caption)
        
        # 保存结果
        df['caption'] = results
        df.to_csv(args.output_path, index=False)
        logger.info(f"Results saved to {args.output_path}")
        
    else:
        # 多GPU并行模式
        num_gpus = torch.cuda.device_count()
        world_size = args.num_workers if args.num_workers else num_gpus
        world_size = min(world_size, num_gpus)
        
        logger.info(f"Running with {world_size} processes on {num_gpus} GPUs")
        
        mp.set_start_method('spawn', force=True)
        
        # 创建输出队列
        output_queue = Queue()
        
        # 启动工作进程
        processes = []
        for rank in range(world_size):
            p = Process(target=worker_process, args=(
                rank, world_size, args.model_id, args.csv_path, 
                output_queue, args
            ))
            p.start()
            processes.append(p)
        
        # 收集结果
        results_by_rank = {}
        for _ in range(world_size):
            rank, results = output_queue.get()
            results_by_rank[rank] = results
            logger.info(f"Received results from process {rank}")
        
        # 等待所有进程结束
        for p in processes:
            p.join()
        
        # 合并结果
        all_results = []
        for rank in range(world_size):
            all_results.extend(results_by_rank[rank])
        
        # 保存结果
        df = pd.read_csv(args.csv_path)
        df['caption'] = all_results
        df.to_csv(args.output_path, index=False)
        logger.info(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
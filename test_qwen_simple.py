#!/usr/bin/env python3
"""
简化的Qwen2.5-VL模型测试脚本，带超时机制
"""
import logging
import signal
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")

def load_simple_video_frames(video_path, num_frames=4):
    """简化的视频帧提取"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 转换BGR到RGB并调整大小
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((224, 224))
            frames.append(pil_image)
        else:
            logger.warning(f"Failed to read frame {frame_idx}")
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def test_simple_generation():
    """简化的生成测试"""
    model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
    video_path = "videos/Dataset/panda/000/---azc0s53U-0:00:37.200-0:00:41.360.mp4"
    
    logger.info("Loading model...")
    
    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={i: "14GiB" for i in range(4)},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    logger.info("Model loaded successfully")
    
    # 加载视频
    logger.info("Loading video frames...")
    frames = load_simple_video_frames(video_path, num_frames=4)
    
    # 构建简化的消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": "Describe this video briefly."}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 移动到第一个GPU
    inputs = inputs.to("cuda:0")
    
    # 设置超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30秒超时
    
    try:
        logger.info("Starting generation with 30s timeout...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,  # 更短的输出
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Generation successful: {output_text}")
            
    except TimeoutError:
        logger.error("Generation timed out after 30 seconds")
        output_text = "Error: Generation timeout"
    except Exception as e:
        logger.error(f"Generation error: {e}")
        output_text = f"Error: {e}"
    finally:
        signal.alarm(0)  # 取消超时
        
    # 清理内存
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    
    print(f"\nFinal output: {output_text}")

if __name__ == "__main__":
    test_simple_generation()
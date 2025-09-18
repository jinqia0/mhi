#!/usr/bin/env python3
"""
深度接触判定快速启动脚本
提供不同的运行模式：测试、小规模处理、大规模处理
"""

import os
import sys
import argparse
import torch
from datetime import datetime


def run_test_mode():
    """运行测试模式"""
    print("🧪 运行测试模式...")
    os.system("python test_parallel_depth.py")


def run_small_batch(max_videos=100):
    """运行小批量处理"""
    print(f"📦 运行小批量处理 (最多 {max_videos} 个视频)...")
    
    # 修改配置
    config_updates = f"""
# 小批量处理配置
config['max_videos'] = {max_videos}
config['chunk_size'] = 50
config['processes_per_gpu'] = 1
config['depth_encoder'] = 'vits'  # 使用最小模型
"""
    
    # 运行处理
    cmd = f"python -c \"exec(open('yolo_depth_parallel.py').read().replace('max_videos': 500', 'max_videos': {max_videos}'))\""
    os.system("python yolo_depth_parallel.py")


def run_full_batch():
    """运行完整批量处理"""
    print("🚀 运行完整批量处理...")
    
    # 检查资源
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("⚠️  警告: 没有检测到GPU，处理速度会很慢")
    else:
        print(f"💪 检测到 {gpu_count} 个GPU，开始大规模处理")
    
    os.system("python yolo_depth_parallel.py")


def run_comparison_analysis():
    """运行对比分析"""
    print("📊 运行对比分析（原始方法 vs 深度方法）...")
    
    # 先运行原始方法
    print("1. 运行原始YOLO分割方法...")
    original_cmd = """
python -c "
import sys
sys.path.append('.')
exec(open('yolo_seg.py').read().replace('contact_stats_main.csv', 'contact_stats_original_comparison.csv'))
"
"""
    os.system(original_cmd)
    
    # 再运行深度方法
    print("2. 运行深度增强方法...")
    os.system("python yolo_depth_parallel.py")
    
    # 生成对比报告
    print("3. 生成对比报告...")
    comparison_script = '''
import pandas as pd
import matplotlib.pyplot as plt

# 读取两个结果
try:
    df_original = pd.read_csv("contact_stats_original_comparison.csv")
    df_depth = pd.read_csv("contact_stats_parallel_depth.csv")
    
    # 统计对比
    total_videos = len(df_original)
    original_contact = len(df_original[df_original["is_contact"] == 1])
    depth_contact = len(df_depth[df_depth["is_depth_contact"] == 1])
    
    print(f"\\n=== 对比分析结果 ===")
    print(f"总视频数: {total_videos}")
    print(f"原始方法检测接触: {original_contact} ({original_contact/total_videos:.3f})")
    print(f"深度方法检测接触: {depth_contact} ({depth_contact/total_videos:.3f})")
    print(f"深度过滤效果: 减少 {original_contact - depth_contact} 个误检 ({(original_contact - depth_contact)/original_contact:.1%})")
    
    # 保存对比报告
    with open("depth_analysis_report.txt", "w") as f:
        f.write(f"深度接触判定对比分析报告\\n")
        f.write(f"生成时间: {datetime.now()}\\n\\n")
        f.write(f"总视频数: {total_videos}\\n")
        f.write(f"原始方法检测接触: {original_contact} ({original_contact/total_videos:.3f})\\n")
        f.write(f"深度方法检测接触: {depth_contact} ({depth_contact/total_videos:.3f})\\n")
        f.write(f"深度过滤效果: 减少 {original_contact - depth_contact} 个误检 ({(original_contact - depth_contact)/original_contact:.1%})\\n")
    
    print("\\n📄 对比报告已保存到: depth_analysis_report.txt")
    
except Exception as e:
    print(f"对比分析失败: {e}")
'''
    
    with open("temp_comparison.py", "w") as f:
        f.write(comparison_script)
    
    os.system("python temp_comparison.py")
    
    # 清理临时文件
    if os.path.exists("temp_comparison.py"):
        os.remove("temp_comparison.py")


def show_status():
    """显示当前状态"""
    print("📋 当前状态检查...")
    
    # 检查输出文件
    output_files = {
        "原始方法结果": "contact_stats_main.csv",
        "深度方法结果": "contact_stats_parallel_depth.csv",
        "对比报告": "depth_analysis_report.txt"
    }
    
    for name, filename in output_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            mtime = datetime.fromtimestamp(os.path.getmtime(filename))
            print(f"✅ {name}: {filename} ({size} bytes, {mtime})")
        else:
            print(f"❌ {name}: {filename} (不存在)")
    
    # 检查临时文件
    temp_dirs = ["temp_depth_processing", "test_temp", "runs"]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            file_count = sum(len(files) for _, _, files in os.walk(temp_dir))
            print(f"🗂️  临时目录: {temp_dir} ({file_count} 个文件)")
    
    # 检查GPU状态
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"🔥 GPU {i}: {memory_used:.1f}/{memory_total:.1f} GB")
    else:
        print("⚠️  GPU: 不可用")


def clean_temp_files():
    """清理临时文件"""
    print("🗑️  清理临时文件...")
    
    import shutil
    
    temp_items = [
        "temp_depth_processing",
        "test_temp", 
        "runs",
        "video_list_temp.csv",
        "test_videos.csv",
        "test_output.csv",
        "temp_comparison.py"
    ]
    
    cleaned_count = 0
    for item in temp_items:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"✅ 已删除: {item}")
                cleaned_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {item}: {e}")
    
    print(f"🧹 清理完成，删除了 {cleaned_count} 个项目")


def main():
    parser = argparse.ArgumentParser(description="深度接触判定分析工具")
    parser.add_argument("mode", choices=[
        "test", "small", "full", "compare", "status", "clean"
    ], help="运行模式")
    parser.add_argument("--max-videos", type=int, default=100, 
                       help="小批量模式的最大视频数量")
    
    args = parser.parse_args()
    
    print(f"🎯 深度接触判定分析工具")
    print(f"模式: {args.mode}")
    print(f"时间: {datetime.now()}")
    print("-" * 50)
    
    if args.mode == "test":
        run_test_mode()
    elif args.mode == "small":
        run_small_batch(args.max_videos)
    elif args.mode == "full":
        run_full_batch()
    elif args.mode == "compare":
        run_comparison_analysis()
    elif args.mode == "status":
        show_status()
    elif args.mode == "clean":
        clean_temp_files()
    
    print("-" * 50)
    print("✨ 完成!")


if __name__ == "__main__":
    main()
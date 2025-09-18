#!/usr/bin/env python3
"""
提取误判视频分析脚本
基于analysis_contact_detection_accuracy.py中的交互判定方法，
从四种判定方式中提取误判的视频路径：
1. 人人接触检测 (is_contact)  
2. 深度接触检测 (is_depth_contact)
3. mask变化检测 (is_mask_contact)  
4. 综合加强检测 (is_any_enhanced_contact)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """加载并合并数据"""
    print("Loading data files...")
    
    # 加载检测结果数据
    detection_df = pd.read_csv('/home/jinqiao/mhi/results/contact_stats_depth_mask_enhanced.csv')
    print(f"Detection data: {len(detection_df)} videos")
    
    # 加载人工标注数据
    manual_df = pd.read_csv('/home/jinqiao/mhi/results/contact_stats_main_panda_manual.csv')
    print(f"Manual annotation data: {len(manual_df)} videos")
    
    # 合并数据（以video路径为key）
    merged_df = pd.merge(detection_df, manual_df[['video', 'manual_annotation']], 
                        on='video', how='inner')
    print(f"Merged data: {len(merged_df)} videos")
    
    return merged_df

def create_mask_contact_column(df):
    """创建mask变化判定列"""
    # 判断是否有较大mask变化 (mask变化率 > 0.2)
    df['is_mask_contact'] = (df['avg_change_ratio'] > 0.2).astype(int)
    return df

def extract_misclassified_videos(df, method_name, prediction_column, output_file):
    """提取某个方法的误判视频"""
    print(f"\n=== Analyzing {method_name} ===")
    
    # 确保数据有效
    valid_mask = ~(pd.isna(df['manual_annotation']) | pd.isna(df[prediction_column]))
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print(f"No valid data for {method_name}")
        return pd.DataFrame()
    
    # 获取真实标签和预测标签
    y_true = valid_df['manual_annotation']
    y_pred = valid_df[prediction_column]
    
    # 计算误判情况
    false_positives = valid_df[(y_true == 0) & (y_pred == 1)]  # 预测为接触，实际无接触
    false_negatives = valid_df[(y_true == 1) & (y_pred == 0)]  # 预测为无接触，实际有接触
    
    print(f"Total valid samples: {len(valid_df)}")
    print(f"False Positives (FP): {len(false_positives)} videos")
    print(f"False Negatives (FN): {len(false_negatives)} videos")
    print(f"Total Misclassified: {len(false_positives) + len(false_negatives)} videos")
    
    # 合并误判视频
    misclassified = pd.concat([
        false_positives.assign(error_type='False Positive'),
        false_negatives.assign(error_type='False Negative')
    ], ignore_index=True)
    
    if len(misclassified) > 0:
        # 保存误判视频信息
        misclassified_info = misclassified[[
            'video', 'manual_annotation', prediction_column, 'error_type'
        ]].copy()
        misclassified_info['detection_method'] = method_name
        misclassified_info.to_csv(output_file, index=False)
        print(f"Misclassified videos saved to: {output_file}")
        
        # 返回误判视频路径列表
        return misclassified[['video', 'error_type']].copy()
    
    return pd.DataFrame()

def find_unique_misclassified_videos(all_misclassified_videos, method_names):
    """找出每种方法独有的误判视频"""
    print("\n=== Finding Unique Misclassified Videos for Each Method ===")
    
    unique_results = {}
    
    for i, method in enumerate(method_names):
        if method not in all_misclassified_videos or len(all_misclassified_videos[method]) == 0:
            unique_results[method] = pd.DataFrame()
            continue
            
        current_method_videos = set(all_misclassified_videos[method]['video'])
        
        # 获取其他方法的误判视频
        other_methods_videos = set()
        for j, other_method in enumerate(method_names):
            if i != j and other_method in all_misclassified_videos:
                other_methods_videos.update(all_misclassified_videos[other_method]['video'])
        
        # 找出当前方法独有的误判视频
        unique_videos = current_method_videos - other_methods_videos
        
        if unique_videos:
            unique_df = all_misclassified_videos[method][
                all_misclassified_videos[method]['video'].isin(unique_videos)
            ].copy()
            unique_results[method] = unique_df
            
            print(f"{method}: {len(unique_videos)} unique misclassified videos")
            
            # 保存独有的误判视频
            output_file = f'/home/jinqiao/mhi/results/{method.lower().replace(" ", "_")}_unique_misclassified.csv'
            unique_df.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")
        else:
            unique_results[method] = pd.DataFrame()
            print(f"{method}: No unique misclassified videos")
    
    return unique_results

def analyze_overlap_patterns(all_misclassified_videos, method_names):
    """分析不同方法间误判视频的重叠模式"""
    print("\n=== Analyzing Overlap Patterns ===")
    
    # 创建视频到方法的映射
    video_to_methods = {}
    
    for method in method_names:
        if method in all_misclassified_videos:
            for _, row in all_misclassified_videos[method].iterrows():
                video = row['video']
                if video not in video_to_methods:
                    video_to_methods[video] = []
                video_to_methods[video].append(method)
    
    # 统计重叠模式
    overlap_patterns = {}
    for video, methods in video_to_methods.items():
        pattern = frozenset(methods)
        if pattern not in overlap_patterns:
            overlap_patterns[pattern] = []
        overlap_patterns[pattern].append(video)
    
    # 输出重叠分析
    overlap_results = []
    for pattern, videos in overlap_patterns.items():
        pattern_str = " + ".join(sorted(pattern))
        overlap_results.append({
            'overlap_pattern': pattern_str,
            'method_count': len(pattern),
            'video_count': len(videos),
            'videos': "; ".join(videos[:5])  # 只显示前5个视频路径
        })
    
    overlap_df = pd.DataFrame(overlap_results)
    overlap_df = overlap_df.sort_values('video_count', ascending=False)
    overlap_df.to_csv('/home/jinqiao/mhi/results/misclassified_overlap_patterns.csv', index=False)
    
    print("Overlap patterns analysis:")
    for _, row in overlap_df.iterrows():
        print(f"  {row['overlap_pattern']}: {row['video_count']} videos")
    
    return overlap_df

def generate_summary_report(df, all_misclassified_videos, unique_results, method_names):
    """生成汇总报告"""
    print("\n=== Generating Summary Report ===")
    
    with open('/home/jinqiao/mhi/results/misclassified_videos_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Misclassified Videos Analysis Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. Dataset Overview\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total videos analyzed: {len(df)}\n")
        f.write(f"Manual annotation distribution:\n")
        f.write(f"  - Contact (1): {sum(df['manual_annotation'])} ({sum(df['manual_annotation'])/len(df)*100:.1f}%)\n")
        f.write(f"  - No Contact (0): {len(df)-sum(df['manual_annotation'])} ({(len(df)-sum(df['manual_annotation']))/len(df)*100:.1f}%)\n\n")
        
        f.write("2. Misclassification Summary by Method\n")
        f.write("-" * 40 + "\n")
        for method in method_names:
            if method in all_misclassified_videos and len(all_misclassified_videos[method]) > 0:
                total_misclassified = len(all_misclassified_videos[method])
                fp_count = len(all_misclassified_videos[method][all_misclassified_videos[method]['error_type'] == 'False Positive'])
                fn_count = len(all_misclassified_videos[method][all_misclassified_videos[method]['error_type'] == 'False Negative'])
                
                f.write(f"{method}:\n")
                f.write(f"  Total misclassified: {total_misclassified} videos\n")
                f.write(f"  False Positives: {fp_count} videos\n")
                f.write(f"  False Negatives: {fn_count} videos\n")
                
                if method in unique_results and len(unique_results[method]) > 0:
                    f.write(f"  Unique misclassified: {len(unique_results[method])} videos\n")
                else:
                    f.write(f"  Unique misclassified: 0 videos\n")
                f.write("\n")
        
        f.write("3. Unique Misclassified Videos by Method\n")
        f.write("-" * 40 + "\n")
        for method in method_names:
            if method in unique_results and len(unique_results[method]) > 0:
                f.write(f"{method} unique misclassified videos:\n")
                for _, row in unique_results[method].iterrows():
                    f.write(f"  - {row['video']} ({row['error_type']})\n")
                f.write("\n")
    
    print("Summary report saved to: /home/jinqiao/mhi/results/misclassified_videos_summary.txt")

def main():
    """主函数"""
    print("Misclassified Videos Extraction Analysis")
    print("=" * 60)
    
    # 加载数据
    df = load_and_merge_data()
    
    # 创建mask变化判定列
    df = create_mask_contact_column(df)
    
    # 定义四种检测方法
    methods = {
        'Human Contact': 'is_contact',
        'Depth Contact': 'is_depth_contact', 
        'Mask Change Contact': 'is_mask_contact',
        'Enhanced Contact': 'is_any_enhanced_contact'
    }
    
    # 创建结果目录
    Path('/home/jinqiao/mhi/results').mkdir(exist_ok=True)
    
    # 分析每种方法的误判视频
    all_misclassified_videos = {}
    
    for method_name, column in methods.items():
        if column in df.columns:
            output_file = f'/home/jinqiao/mhi/results/{method_name.lower().replace(" ", "_")}_misclassified.csv'
            misclassified = extract_misclassified_videos(df, method_name, column, output_file)
            all_misclassified_videos[method_name] = misclassified
        else:
            print(f"Warning: Column '{column}' not found for {method_name}")
            all_misclassified_videos[method_name] = pd.DataFrame()
    
    # 找出每种方法独有的误判视频
    method_names = list(methods.keys())
    unique_results = find_unique_misclassified_videos(all_misclassified_videos, method_names)
    
    # 分析重叠模式
    overlap_df = analyze_overlap_patterns(all_misclassified_videos, method_names)
    
    # 生成汇总报告
    generate_summary_report(df, all_misclassified_videos, unique_results, method_names)
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("Generated files:")
    for method_name in method_names:
        method_file = method_name.lower().replace(" ", "_")
        print(f"  - /home/jinqiao/mhi/results/{method_file}_misclassified.csv")
        print(f"  - /home/jinqiao/mhi/results/{method_file}_unique_misclassified.csv")
    print("  - /home/jinqiao/mhi/results/misclassified_overlap_patterns.csv")
    print("  - /home/jinqiao/mhi/results/misclassified_videos_summary.txt")

if __name__ == "__main__":
    main()
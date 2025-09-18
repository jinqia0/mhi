#!/usr/bin/env python3
"""
Extract Enhancement Effect Samples

This script extracts specific video samples to demonstrate the direct effects of each enhancement method:
1. Correctly identified new positives (False Negatives → True Positives)
2. Incorrectly identified new positives (True Negatives → False Positives)  
3. Lost true positives (True Positives → False Negatives)

Videos are copied to organized folders for manual inspection.
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

def load_data():
    """Load detection results and manual annotations"""
    print("Loading data...")
    
    # Load detection results with enhanced methods
    detection_df = pd.read_csv('/home/jinqiao/mhi/results/contact_stats_depth_mask_enhanced.csv')
    
    # Load manual annotations (ground truth)
    manual_df = pd.read_csv('/home/jinqiao/mhi/results/contact_stats_main_panda_manual.csv')
    
    # Merge on video column
    merged_df = pd.merge(detection_df, manual_df[['video', 'manual_annotation']], 
                        on='video', how='inner')
    
    print(f"Loaded {len(merged_df)} videos with both detection results and manual annotations")
    
    return merged_df

def create_progressive_methods(df):
    """Create progressive detection methods"""
    print("Creating progressive detection methods...")
    
    # Ground truth
    y_true = df['manual_annotation'].values
    
    # Base method: Segmentation overlap detection
    base_segmentation = df['is_contact'].fillna(0).astype(int)
    
    # Progressive enhancements
    depth_enhanced = df['is_depth_contact'].fillna(0).astype(int)
    
    # For mask change enhancement
    mask_change_threshold = df['avg_change_ratio'].median()
    mask_change_binary = (df['avg_change_ratio'] > mask_change_threshold).astype(int)
    mask_change_enhanced = base_segmentation & mask_change_binary
    
    # Indirect contact
    indirect_contact = df['is_indirect_contact'].fillna(0).astype(int)
    base_plus_indirect = base_segmentation | indirect_contact
    
    # Combined enhancement
    combined_enhanced = df['is_any_enhanced_contact'].fillna(0).astype(int)
    
    methods = {
        'base': base_segmentation,
        'depth': depth_enhanced,
        'mask_change': mask_change_enhanced, 
        'indirect': base_plus_indirect,
        'combined': combined_enhanced
    }
    
    return methods, y_true

def create_output_directories(base_output_dir='/home/jinqiao/mhi/enhancement_samples'):
    """Create organized output directory structure"""
    print(f"Creating output directories in {base_output_dir}...")
    
    base_path = Path(base_output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Enhancement methods
    methods = ['depth', 'mask_change', 'indirect', 'combined']
    
    # Effect types
    effect_types = [
        'new_correct_positives',    # FN → TP (good improvement)
        'new_correct_negatives',    # FP → TN (good improvement - correctly excluded false positives)
        'new_incorrect_positives',  # TN → FP (bad side effect)
        'lost_true_positives'       # TP → FN (bad degradation)
    ]
    
    dirs_created = {}
    
    for method in methods:
        dirs_created[method] = {}
        method_path = base_path / method
        method_path.mkdir(exist_ok=True)
        
        for effect_type in effect_types:
            effect_path = method_path / effect_type
            effect_path.mkdir(exist_ok=True)
            dirs_created[method][effect_type] = effect_path
    
    return dirs_created

def analyze_enhancement_effects(df, methods, y_true):
    """Analyze the specific effects of each enhancement method"""
    print("Analyzing enhancement effects...")
    
    base_pred = methods['base']
    results = {}
    
    enhancement_methods = {
        'depth': 'Base + Depth Enhancement',
        'mask_change': 'Base + Mask Change Enhancement', 
        'indirect': 'Base + Indirect Contact',
        'combined': 'All Combined'
    }
    
    for method_key, method_name in enhancement_methods.items():
        enhanced_pred = methods[method_key]
        
        print(f"\n=== Analyzing {method_name} ===")
        
        # Find different types of changes
        # 1. New Correct Positives: Base=0, Enhanced=1, Truth=1 (FN → TP)
        new_correct_positives = (base_pred == 0) & (enhanced_pred == 1) & (y_true == 1)
        
        # 2. New Correct Negatives: Base=1, Enhanced=0, Truth=0 (FP → TN) - GOOD!
        new_correct_negatives = (base_pred == 1) & (enhanced_pred == 0) & (y_true == 0)
        
        # 3. New Incorrect Positives: Base=0, Enhanced=1, Truth=0 (TN → FP)  
        new_incorrect_positives = (base_pred == 0) & (enhanced_pred == 1) & (y_true == 0)
        
        # 4. Lost True Positives: Base=1, Enhanced=0, Truth=1 (TP → FN)
        lost_true_positives = (base_pred == 1) & (enhanced_pred == 0) & (y_true == 1)
        
        # Get video indices for each category
        new_correct_positives_indices = df.index[new_correct_positives].tolist()
        new_correct_negatives_indices = df.index[new_correct_negatives].tolist()
        new_incorrect_positives_indices = df.index[new_incorrect_positives].tolist()
        lost_true_positives_indices = df.index[lost_true_positives].tolist()
        
        results[method_key] = {
            'new_correct_positives': new_correct_positives_indices,
            'new_correct_negatives': new_correct_negatives_indices,
            'new_incorrect_positives': new_incorrect_positives_indices,
            'lost_true_positives': lost_true_positives_indices
        }
        
        print(f"  新增正确阳性 (FN→TP): {len(new_correct_positives_indices)}")
        print(f"  新增正确阴性 (FP→TN): {len(new_correct_negatives_indices)}")
        print(f"  新增错误阳性 (TN→FP): {len(new_incorrect_positives_indices)}")
        print(f"  丢失真实阳性 (TP→FN): {len(lost_true_positives_indices)}")
        
        # Show some examples
        if len(new_correct_positives_indices) > 0:
            print("  新增正确阳性示例:")
            for idx in new_correct_positives_indices[:3]:
                video_name = df.loc[idx, 'video'].split('/')[-1]
                print(f"    - {video_name}")
        
        if len(new_correct_negatives_indices) > 0:
            print("  新增正确阴性示例 (正确排除的非接触视频):")
            for idx in new_correct_negatives_indices[:3]:
                video_name = df.loc[idx, 'video'].split('/')[-1]
                print(f"    - {video_name}")
        
        if len(new_incorrect_positives_indices) > 0:
            print("  新增错误阳性示例:")
            for idx in new_incorrect_positives_indices[:3]:
                video_name = df.loc[idx, 'video'].split('/')[-1]
                print(f"    - {video_name}")
                
        if len(lost_true_positives_indices) > 0:
            print("  丢失真实阳性示例:")
            for idx in lost_true_positives_indices[:3]:
                video_name = df.loc[idx, 'video'].split('/')[-1]
                print(f"    - {video_name}")
    
    return results

def copy_sample_videos(df, results, dirs_created, max_samples_per_category=10):
    """Copy sample videos to organized directories"""
    print("\nCopying sample videos to organized directories...")
    
    total_copied = 0
    copy_log = []
    
    for method_key, method_results in results.items():
        print(f"\n--- Processing {method_key} method ---")
        
        for effect_type, video_indices in method_results.items():
            if len(video_indices) == 0:
                print(f"  No videos found for {effect_type}")
                continue
            
            # Limit number of samples to copy
            samples_to_copy = min(len(video_indices), max_samples_per_category)
            selected_indices = video_indices[:samples_to_copy]
            
            target_dir = dirs_created[method_key][effect_type]
            
            print(f"  Copying {samples_to_copy} videos to {effect_type}...")
            
            copied_count = 0
            for idx in selected_indices:
                video_path = df.loc[idx, 'video']
                video_name = os.path.basename(video_path)
                
                # Check if source video exists
                if os.path.exists(video_path):
                    target_path = target_dir / video_name
                    
                    try:
                        shutil.copy2(video_path, target_path)
                        copied_count += 1
                        total_copied += 1
                        
                        # Log the copy operation
                        copy_log.append({
                            'method': method_key,
                            'effect_type': effect_type,
                            'video_name': video_name,
                            'source_path': video_path,
                            'target_path': str(target_path),
                            'manual_annotation': df.loc[idx, 'manual_annotation'],
                            'base_prediction': df.loc[idx, 'is_contact'],
                            'enhanced_prediction': get_enhanced_prediction(df, idx, method_key)
                        })
                        
                    except Exception as e:
                        print(f"    Error copying {video_name}: {e}")
                else:
                    print(f"    Warning: Source video not found: {video_path}")
            
            print(f"    Successfully copied {copied_count}/{samples_to_copy} videos")
    
    print(f"\nTotal videos copied: {total_copied}")
    
    # Save copy log
    if copy_log:
        log_df = pd.DataFrame(copy_log)
        log_df.to_csv('/home/jinqiao/mhi/enhancement_samples/copy_log.csv', index=False)
        print("Copy log saved to: /home/jinqiao/mhi/enhancement_samples/copy_log.csv")
    
    return copy_log

def get_enhanced_prediction(df, idx, method_key):
    """Get the enhanced prediction for a specific method"""
    if method_key == 'depth':
        return df.loc[idx, 'is_depth_contact']
    elif method_key == 'mask_change':
        base = df.loc[idx, 'is_contact']
        mask_change = df.loc[idx, 'avg_change_ratio'] > df['avg_change_ratio'].median()
        return int(base & mask_change)
    elif method_key == 'indirect':
        base = df.loc[idx, 'is_contact'] 
        indirect = df.loc[idx, 'is_indirect_contact']
        return int(base | indirect)
    elif method_key == 'combined':
        return df.loc[idx, 'is_any_enhanced_contact']
    else:
        return 0

def generate_summary_report(results, copy_log, base_output_dir):
    """Generate a summary report of extracted samples"""
    print("Generating summary report...")
    
    report = []
    report.append("# Enhancement Effect Sample Extraction Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("## Overview")
    report.append("This report summarizes video samples extracted to demonstrate the effects of different enhancement methods.")
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    method_names = {
        'depth': 'Depth Enhancement',
        'mask_change': 'Mask Change Enhancement',
        'indirect': 'Indirect Contact Enhancement', 
        'combined': 'Combined Enhancement'
    }
    
    for method_key, method_name in method_names.items():
        if method_key in results:
            method_results = results[method_key]
            report.append(f"### {method_name}")
            report.append(f"- 新增正确阳性 (FN→TP): {len(method_results['new_correct_positives'])} 个视频")
            report.append(f"- 新增正确阴性 (FP→TN): {len(method_results['new_correct_negatives'])} 个视频")
            report.append(f"- 新增错误阳性 (TN→FP): {len(method_results['new_incorrect_positives'])} 个视频")
            report.append(f"- 丢失真实阳性 (TP→FN): {len(method_results['lost_true_positives'])} 个视频")
            
            net_positive_effect = len(method_results['new_correct_positives']) - len(method_results['new_incorrect_positives']) - len(method_results['lost_true_positives'])
            net_negative_effect = len(method_results['new_correct_negatives'])
            report.append(f"- **净正效应**: {net_positive_effect:+} 个视频")
            report.append(f"- **净负效应**: +{net_negative_effect} 个正确排除的视频")
            report.append("")
    
    # Directory structure
    report.append("## Directory Structure")
    report.append("")
    report.append("```")
    report.append("enhancement_samples/")
    for method_key in method_names.keys():
        report.append(f"├── {method_key}/")
        report.append(f"│   ├── new_correct_positives/     # 此增强方法正确识别的接触视频")
        report.append(f"│   ├── new_correct_negatives/     # 此增强方法正确排除的非接触视频")
        report.append(f"│   ├── new_incorrect_positives/   # 此增强方法错误识别的视频") 
        report.append(f"│   └── lost_true_positives/       # 此增强方法遗漏的真实接触视频")
    report.append("└── copy_log.csv                      # Detailed log of all copied videos")
    report.append("```")
    report.append("")
    
    # Analysis recommendations
    report.append("## Analysis Recommendations")
    report.append("")
    report.append("### 人工审查建议:")
    report.append("1. **new_correct_positives/**: 观看这些视频了解每种增强方法成功捕获的模式")
    report.append("2. **new_correct_negatives/**: 观看这些视频了解每种增强方法正确排除的非接触模式")
    report.append("3. **new_incorrect_positives/**: 观看这些视频了解每种增强方法产生误报的原因")  
    report.append("4. **lost_true_positives/**: 观看这些视频了解哪些真实接触被过滤掉了")
    report.append("")
    report.append("### 需要回答的关键问题:")
    report.append("- 正确识别和错误识别的样本有什么视觉特征区别？")
    report.append("- 不同增强方法是否存在共同的失败模式？")
    report.append("- 能否根据观察到的模式调整阈值或逻辑？")
    report.append("- 哪种类型的非接触场景最容易被错误识别为接触？")
    report.append("")
    
    # Save report
    report_path = Path(base_output_dir) / "extraction_summary_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to: {report_path}")
    
    return report

def main():
    """Main extraction function"""
    print("Enhancement Effect Sample Extraction")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Create progressive detection methods
    methods, y_true = create_progressive_methods(df)
    
    # Create output directory structure
    dirs_created = create_output_directories()
    
    # Analyze enhancement effects
    results = analyze_enhancement_effects(df, methods, y_true)
    
    # Copy sample videos to organized directories
    copy_log = copy_sample_videos(df, results, dirs_created)
    
    # Generate summary report
    summary_report = generate_summary_report(results, copy_log, '/home/jinqiao/mhi/enhancement_samples')
    
    print("\n=== Sample Extraction Complete ===")
    print("Generated structure:")
    print("- /home/jinqiao/mhi/enhancement_samples/")
    print("  ├── depth/                        # 深度增强效果样本")
    print("  ├── mask_change/                  # 掩码变化增强效果样本") 
    print("  ├── indirect/                     # 间接接触增强效果样本")
    print("  ├── combined/                     # 组合增强效果样本")
    print("  ├── copy_log.csv                  # 详细复制日志")
    print("  └── extraction_summary_report.txt # 提取总结报告")
    
    print(f"\n提取用于人工审查的视频总数: {len(copy_log) if copy_log else 0}")
    print("\n现在可以人工审查各类别中的视频，以了解每种增强方法的具体效果！")
    print("\n特别关注:")
    print("- new_correct_negatives/ 文件夹中正确排除的非接触视频")
    print("- 这些视频展示了增强方法如何改进假阳性问题")

if __name__ == "__main__":
    main()
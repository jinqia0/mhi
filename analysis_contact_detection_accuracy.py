#!/usr/bin/env python3
"""
Contact Detection Accuracy Analysis
分析不同人人接触检测方法的准确率统计

将manual_annotation列视作真值，分析以下检测方法的性能：
1. 分割点相连检测 (is_contact)  
2. 深度判定 (is_depth_contact)
3. mask变化判定 (is_any_enhanced_contact)  # 使用新列名
4. 综合加强判定 (is_any_enhanced_contact)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def calculate_metrics(y_true, y_pred, method_name):
    """计算分类指标"""
    # 过滤掉NaN值
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'method': method_name,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'samples': 0
        }
    
    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    precision = precision_score(y_true_clean, y_pred_clean, zero_division=0)
    recall = recall_score(y_true_clean, y_pred_clean, zero_division=0)
    f1 = f1_score(y_true_clean, y_pred_clean, zero_division=0)
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'samples': len(y_true_clean)
    }

def create_combination_methods(df):
    """创建三处加强的组合方法"""
    print("\n=== Creating Combination Detection Methods ===")

    # 判断是否有较大mask变化 (mask变化率 > 0.2)，创建 is_mask_contact 列
    df['is_mask_contact'] = (df['avg_change_ratio'] > 0.2).astype(int)

    # 添加 is_any_enhanced_contact 为新的接触判定条件
    df['is_any_enhanced_contact'] = df['is_any_enhanced_contact'].fillna(0).astype(int)  # 使用新列名
    
    # 创建组合判定
    # OR组合 (任一为真)
    df['combo_any_3'] = ((df['is_contact'] == 1) | 
                         (df['is_depth_contact'] == 1) | 
                         (df['is_mask_contact'] == 1) |
                         (df['is_any_enhanced_contact'] == 1)).astype(int)
    
    # AND组合 (全部为真)
    df['combo_all_3'] = ((df['is_contact'] == 1) & 
                         (df['is_depth_contact'] == 1) & 
                         (df['is_mask_contact'] == 1) &
                         (df['is_any_enhanced_contact'] == 1)).astype(int)

    return df

def analyze_detection_methods(df):
    """分析各种检测方法的性能"""
    print("\n=== Detection Methods Performance Analysis ===")
    
    # 创建组合方法
    df = create_combination_methods(df)

    # 定义所有检测方法
    methods = {
        # 基础方法
        'Segmentation Contact': 'is_contact',
        'Depth Contact': 'is_depth_contact', 
        'Mask Change Contact': 'is_mask_contact',
        'Enhanced Contact': 'is_any_enhanced_contact',  # 新增的 enhanced contact 方法
        
        # 综合判定 (OR组合)
        'Any 3 Methods (OR)': 'combo_any_3',
        
        # 综合判定 (AND组合)
        'All 3 Methods (AND)': 'combo_all_3',
    }

    results = []
    y_true = df['manual_annotation'].values

    print(f"\nGround Truth Distribution:")
    print(f"Total samples: {len(df)}")
    print(f"Contact (1): {sum(y_true)} ({sum(y_true)/len(y_true)*100:.1f}%)")
    print(f"No Contact (0): {len(y_true)-sum(y_true)} ({(len(y_true)-sum(y_true))/len(y_true)*100:.1f}%)")

    for method_name, column in methods.items():
        if column in df.columns:
            y_pred = df[column].values
            metrics = calculate_metrics(y_true, y_pred, method_name)
            results.append(metrics)

            print(f"\n{method_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")
            print(f"  Samples:   {metrics['samples']}")

            # 计算混淆矩阵
            mask = ~(pd.isna(y_true) | pd.isna(y_pred))
            if mask.sum() > 0:
                cm = confusion_matrix(y_true[mask], y_pred[mask])
                print(f"  Confusion Matrix:")
                print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
                print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        else:
            print(f"\nWarning: Column '{column}' not found for {method_name}")

    return pd.DataFrame(results), df


def plot_performance_comparison(results_df):
    """绘制性能对比图表"""
    print("\n=== Generating Performance Comparison Plots ===")
    
    # 创建更大的图表以容纳更多方法
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Contact Detection Methods Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # 为不同类型的方法分配颜色
    colors = []
    for method in results_df['method']:
        if 'OR' in method or 'Any' in method:
            colors.append('#ff7f0e')  # 橙色 - OR组合
        elif 'AND' in method or 'All' in method:
            colors.append('#d62728')  # 红色 - AND组合
        elif 'Majority' in method:
            colors.append('#9467bd')  # 紫色 - 投票方法
        elif 'Enhanced' in method:
            colors.append('#2ca02c')  # 绿色 - 原有增强方法
        else:
            colors.append('#1f77b4')  # 蓝色 - 基础方法
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        bars = ax.bar(range(len(results_df)), results_df[metric], 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(name, fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 设置x轴标签
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df['method'], rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/jinqiao/mhi/results/contact_detection_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(df):
    """绘制混淆矩阵对比"""
    print("\n=== Generating Confusion Matrices ===")
    
    methods = {
        'Segmentation Contact': 'is_contact',
        'Depth Contact': 'is_depth_contact', 
        'Mask Change Contact': 'is_any_enhanced_contact',
        'Enhanced Combined': 'is_any_enhanced_contact'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for Different Detection Methods', fontsize=16, fontweight='bold')
    
    y_true = df['manual_annotation'].values
    
    for i, (method_name, column) in enumerate(methods.items()):
        ax = axes[i//2, i%2]
        
        if column in df.columns:
            y_pred = df[column].values
            
            # 过滤掉NaN值
            mask = ~(pd.isna(y_true) | pd.isna(y_pred))
            if mask.sum() > 0:
                cm = confusion_matrix(y_true[mask], y_pred[mask])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['No Contact', 'Contact'],
                           yticklabels=['No Contact', 'Contact'])
                ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Column {column}\nNot Found', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/jinqiao/mhi/results/contact_detection_confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(df, results_df):
    """生成详细的分析报告"""
    print("\n=== Generating Detailed Analysis Report ===")
    
    with open('/home/jinqiao/mhi/results/contact_detection_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("Contact Detection Methods Performance Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. Dataset Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total videos analyzed: {len(df)}\n")
        f.write(f"Manual annotation distribution:\n")
        f.write(f"  - Contact (1): {sum(df['manual_annotation'])} ({sum(df['manual_annotation'])/len(df)*100:.1f}%)\n")
        f.write(f"  - No Contact (0): {len(df)-sum(df['manual_annotation'])} ({(len(df)-sum(df['manual_annotation']))/len(df)*100:.1f}%)\n\n")
        
        f.write("2. Detection Methods Performance\n")
        f.write("-" * 35 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['method']}:\n")
            f.write(f"  Accuracy:  {row['accuracy']:.4f}\n")
            f.write(f"  Precision: {row['precision']:.4f}\n")
            f.write(f"  Recall:    {row['recall']:.4f}\n")
            f.write(f"  F1-Score:  {row['f1_score']:.4f}\n")
            f.write(f"  Samples:   {row['samples']}\n\n")
        
        f.write("3. Key Findings\n")
        f.write("-" * 15 + "\n")
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        
        f.write(f"Best Accuracy: {best_accuracy['method']} ({best_accuracy['accuracy']:.4f})\n")
        f.write(f"Best F1-Score: {best_f1['method']} ({best_f1['f1_score']:.4f})\n\n")
        
        f.write("4. Method Comparison\n")
        f.write("-" * 20 + "\n")
        sorted_results = results_df.sort_values('f1_score', ascending=False)
        f.write("Ranking by F1-Score:\n")
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            f.write(f"  {i}. {row['method']}: {row['f1_score']:.4f}\n")
    
    print("Report saved to: /home/jinqiao/mhi/results/contact_detection_analysis_report.txt")

def create_combination_summary_table(df):
    """创建组合方法的汇总统计表"""
    print("\n=== Creating Combination Methods Summary ===")
    
    methods = {
        'Segmentation': 'is_contact',
        'Depth': 'is_depth_contact', 
        'Mask Change': 'is_any_enhanced_contact',
        'Seg OR Depth': 'combo_seg_depth',
        'Seg OR Mask': 'combo_seg_mask',
        'Depth OR Mask': 'combo_depth_mask', 
        'Any 3 Methods': 'combo_any_3',
        'Seg AND Depth': 'combo_all_2',
        'All 3 Methods': 'combo_all_3',
        'Majority Vote': 'combo_majority',
        'Enhanced Combined': 'is_any_enhanced_contact'
    }
    
    summary_data = []
    y_true = df['manual_annotation'].values
    
    for method_name, column in methods.items():
        if column in df.columns:
            y_pred = df[column].values
            
            # 统计预测结果分布
            contact_pred = sum(y_pred)
            no_contact_pred = len(y_pred) - contact_pred
            
            # 计算与真值的重叠情况
            true_positives = sum((y_true == 1) & (y_pred == 1))
            false_positives = sum((y_true == 0) & (y_pred == 1))
            false_negatives = sum((y_true == 1) & (y_pred == 0))
            true_negatives = sum((y_true == 0) & (y_pred == 0))
            
            summary_data.append({
                'Method': method_name,
                'Predicted Contact': contact_pred,
                'Predicted No Contact': no_contact_pred,
                'True Positives': true_positives,
                'False Positives': false_positives,
                'False Negatives': false_negatives,
                'True Negatives': true_negatives,
                'Contact Prediction Rate': f"{contact_pred/len(y_pred)*100:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('/home/jinqiao/mhi/results/contact_detection_combination_summary.csv', index=False)
    print("Combination methods summary saved to: /home/jinqiao/mhi/results/contact_detection_combination_summary.csv")
    
    # 打印汇总表
    print("\nCombination Methods Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """主函数"""
    print("Contact Detection Accuracy Analysis")
    print("=" * 50)
    
    # 加载数据
    df = load_and_merge_data()
    
    # 分析检测方法性能
    results_df, enhanced_df = analyze_detection_methods(df)
    
    # 创建组合方法汇总表
    summary_df = create_combination_summary_table(enhanced_df)
    
    # 保存结果表格
    results_df.to_csv('/home/jinqiao/mhi/results/contact_detection_performance_metrics.csv', index=False)
    print(f"\nPerformance metrics saved to: /home/jinqiao/mhi/results/contact_detection_performance_metrics.csv")
    
    # 生成可视化图表
    plot_performance_comparison(results_df)
    plot_confusion_matrices(enhanced_df)
    
    # 生成详细报告
    generate_detailed_report(enhanced_df, results_df)
    
    print("\n" + "=" * 50)
    print("Analysis completed successfully!")
    print("Generated files:")
    print("  - /home/jinqiao/mhi/results/contact_detection_performance_metrics.csv")
    print("  - /home/jinqiao/mhi/results/contact_detection_combination_summary.csv")
    print("  - /home/jinqiao/mhi/results/contact_detection_performance_comparison.png")
    print("  - /home/jinqiao/mhi/results/contact_detection_confusion_matrices.png")
    print("  - /home/jinqiao/mhi/results/contact_detection_analysis_report.txt")

if __name__ == "__main__":
    main()

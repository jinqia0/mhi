#!/usr/bin/env python3
"""
Progressive Enhancement Analysis for Contact Detection

This script analyzes the progressive enhancement effects:
1. Base: Segmentation Overlap Detection (is_contact)
2. Enhancement 1: Base + Depth Consistency (is_depth_contact) 
3. Enhancement 2: Base + Mask Change Analysis (avg_change_ratio threshold)
4. Enhancement 3: Base + Indirect Contact (is_indirect_contact)
5. Combined: All enhancements together (is_any_enhanced_contact)

Key insight: Enhancements 1 & 2 are applied ON TOP of base segmentation detection,
not as independent methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

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

def create_progressive_detection_methods(df):
    """Create progressive detection methods based on enhancement logic"""
    print("\n=== Creating Progressive Detection Methods ===")
    
    # Ground truth
    y_true = df['manual_annotation'].values
    
    # Base method: Segmentation overlap detection
    base_segmentation = df['is_contact'].fillna(0).astype(int)
    
    # Progressive enhancements (applied on top of base segmentation)
    depth_enhanced = df['is_depth_contact'].fillna(0).astype(int)
    
    # For mask change, we need to determine threshold and apply it on top of segmentation
    mask_change_threshold = df['avg_change_ratio'].median()
    mask_change_binary = (df['avg_change_ratio'] > mask_change_threshold).astype(int)
    # Apply mask change enhancement only where base segmentation detected contact
    mask_change_enhanced = base_segmentation & mask_change_binary
    
    # Indirect contact (independent detection)
    indirect_contact = df['is_indirect_contact'].fillna(0).astype(int)
    
    # Combined enhancement (all methods)
    combined_enhanced = df['is_any_enhanced_contact'].fillna(0).astype(int)
    
    # Create progressive combinations
    base_plus_depth = base_segmentation | depth_enhanced
    base_plus_mask = base_segmentation | mask_change_enhanced  
    base_plus_indirect = base_segmentation | indirect_contact
    
    methods = {
        'Base Segmentation': base_segmentation,
        'Base + Depth Enhancement': depth_enhanced,  # This IS base + depth
        'Base + Mask Change Enhancement': mask_change_enhanced,
        'Base + Indirect Contact': base_plus_indirect,
        'All Combined': combined_enhanced
    }
    
    print(f"Using mask change threshold: {mask_change_threshold:.4f}")
    
    return methods, y_true

def compute_progressive_metrics(methods, y_true):
    """Compute metrics for each progressive method"""
    print("\n=== Computing Progressive Enhancement Metrics ===")
    
    results = []
    
    for method_name, y_pred in methods.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        result = {
            'Method': method_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'True Positives': tp,
            'False Positives': fp,
            'False Negatives': fn,
            'True Negatives': tn,
            'Total Positive Predictions': tp + fp,
            'Positive Rate': (tp + fp) / len(y_true)
        }
        
        results.append(result)
        
        print(f"\n{method_name}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return pd.DataFrame(results)

def analyze_enhancement_effects(results_df):
    """Analyze the effect of each enhancement over the base method"""
    print("\n=== Enhancement Effect Analysis ===")
    
    base_performance = results_df[results_df['Method'] == 'Base Segmentation'].iloc[0]
    
    print(f"Base Method Performance:")
    print(f"  F1-Score: {base_performance['F1-Score']:.3f}")
    print(f"  Recall: {base_performance['Recall']:.3f}")
    print(f"  Precision: {base_performance['Precision']:.3f}")
    
    print(f"\nEnhancement Effects:")
    for _, row in results_df.iterrows():
        if row['Method'] != 'Base Segmentation':
            f1_change = row['F1-Score'] - base_performance['F1-Score']
            recall_change = row['Recall'] - base_performance['Recall']
            precision_change = row['Precision'] - base_performance['Precision']
            fp_change = row['False Positives'] - base_performance['False Positives']
            fn_change = row['False Negatives'] - base_performance['False Negatives']
            
            print(f"\n{row['Method']}:")
            print(f"  F1-Score change: {f1_change:+.3f}")
            print(f"  Recall change: {recall_change:+.3f}")
            print(f"  Precision change: {precision_change:+.3f}")
            print(f"  False Positives change: {fp_change:+}")
            print(f"  False Negatives change: {fn_change:+}")

def analyze_enhancement_overlap(df, methods, y_true):
    """Analyze how enhancements affect different types of videos"""
    print("\n=== Enhancement Overlap Analysis ===")
    
    base_segmentation = methods['Base Segmentation']
    depth_enhanced = methods['Base + Depth Enhancement']
    combined_enhanced = methods['All Combined']
    
    # Videos where base method failed but enhancement succeeded
    base_wrong = (base_segmentation != y_true)
    enhanced_right = (combined_enhanced == y_true)
    
    improvement_cases = base_wrong & enhanced_right
    degradation_cases = ~base_wrong & (combined_enhanced != y_true)
    
    print(f"Videos where enhancements improved detection: {improvement_cases.sum()}")
    print(f"Videos where enhancements degraded detection: {degradation_cases.sum()}")
    
    if improvement_cases.sum() > 0:
        print(f"\nExamples of improved detection:")
        improved_videos = df[improvement_cases]
        for i, (_, row) in enumerate(improved_videos.head(5).iterrows()):
            video_name = row['video'].split('/')[-1]
            print(f"  {i+1}. {video_name}")
            print(f"     Base: {base_segmentation[row.name]}, Enhanced: {combined_enhanced[row.name]}, Truth: {y_true[row.name]}")
    
    if degradation_cases.sum() > 0:
        print(f"\nExamples of degraded detection:")
        degraded_videos = df[degradation_cases]
        for i, (_, row) in enumerate(degraded_videos.head(5).iterrows()):
            video_name = row['video'].split('/')[-1]
            print(f"  {i+1}. {video_name}")
            print(f"     Base: {base_segmentation[row.name]}, Enhanced: {combined_enhanced[row.name]}, Truth: {y_true[row.name]}")

def create_progressive_visualizations(results_df, df, methods, y_true):
    """Create visualizations for progressive enhancement analysis"""
    print("\n=== Creating Progressive Enhancement Visualizations ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Progressive Performance Metrics
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)
    
    plt.xlabel('Enhancement Methods')
    plt.ylabel('Score')
    plt.title('Progressive Enhancement Performance')
    plt.xticks(x + width*1.5, results_df['Method'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1-Score Progression
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(range(len(results_df)), results_df['F1-Score'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Enhancement Stage')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Progression')
    plt.xticks(range(len(results_df)), [m.split()[-1] if len(m.split()) > 1 else m for m in results_df['Method']], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. False Positives vs False Negatives
    ax3 = plt.subplot(3, 3, 3)
    plt.scatter(results_df['False Positives'], results_df['False Negatives'], 
                s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
    for i, method in enumerate(results_df['Method']):
        plt.annotate(method.split()[-1] if len(method.split()) > 1 else method, 
                    (results_df['False Positives'].iloc[i], results_df['False Negatives'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('False Positives')
    plt.ylabel('False Negatives')
    plt.title('Error Type Trade-off')
    plt.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall
    ax4 = plt.subplot(3, 3, 4)
    plt.plot(results_df['Recall'], results_df['Precision'], 'o-', linewidth=2, markersize=8)
    for i, method in enumerate(results_df['Method']):
        plt.annotate(f"{i+1}", 
                    (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # 5. Positive Prediction Rate
    ax5 = plt.subplot(3, 3, 5)
    plt.bar(range(len(results_df)), results_df['Positive Rate'], alpha=0.7)
    plt.xlabel('Enhancement Methods')
    plt.ylabel('Positive Prediction Rate')
    plt.title('Positive Prediction Rate by Method')
    plt.xticks(range(len(results_df)), [m.split()[-1] if len(m.split()) > 1 else m for m in results_df['Method']], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Confusion Matrix Heatmap for Base Method
    ax6 = plt.subplot(3, 3, 6)
    base_method = 'Base Segmentation'
    base_idx = results_df[results_df['Method'] == base_method].index[0]
    base_result = results_df.iloc[base_idx]
    base_cm = np.array([[base_result['True Negatives'], base_result['False Positives']],
                       [base_result['False Negatives'], base_result['True Positives']]])
    sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['True Neg', 'True Pos'])
    plt.title(f'{base_method} Confusion Matrix')
    
    # 7. Confusion Matrix Heatmap for Best Enhanced Method
    ax7 = plt.subplot(3, 3, 7)
    best_method_idx = results_df['F1-Score'].idxmax()
    best_result = results_df.iloc[best_method_idx]
    best_cm = np.array([[best_result['True Negatives'], best_result['False Positives']],
                       [best_result['False Negatives'], best_result['True Positives']]])
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['True Neg', 'True Pos'])
    plt.title(f'{best_result["Method"]} Confusion Matrix')
    
    # 8. Enhancement Effect Comparison
    ax8 = plt.subplot(3, 3, 8)
    base_f1 = results_df[results_df['Method'] == 'Base Segmentation']['F1-Score'].iloc[0]
    f1_improvements = results_df['F1-Score'] - base_f1
    colors = ['red' if x < 0 else 'green' for x in f1_improvements]
    plt.bar(range(len(f1_improvements)), f1_improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Enhancement Methods')
    plt.ylabel('F1-Score Change vs Base')
    plt.title('Enhancement Effect on F1-Score')
    plt.xticks(range(len(results_df)), [m.split()[-1] if len(m.split()) > 1 else m for m in results_df['Method']], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 9. Method Performance Radar Chart
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    
    # Select a few methods for radar chart
    methods_for_radar = ['Base Segmentation', 'Base + Depth Enhancement', 'All Combined']
    metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    for method in methods_for_radar:
        if method in results_df['Method'].values:
            method_data = results_df[results_df['Method'] == method].iloc[0]
            values = [method_data[metric] for metric in metrics_for_radar]
            values += [values[0]]  # Complete the circle
            ax9.plot(angles, values, 'o-', linewidth=2, label=method)
            ax9.fill(angles, values, alpha=0.1)
    
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics_for_radar)
    ax9.set_ylim(0, 1)
    plt.title('Method Performance Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.savefig('/home/jinqiao/mhi/results/progressive_enhancement_analysis.png', dpi=300, bbox_inches='tight')
    print("Progressive enhancement visualization saved to: /home/jinqiao/mhi/results/progressive_enhancement_analysis.png")
    
    return fig

def generate_progressive_report(results_df, df, methods, y_true):
    """Generate comprehensive progressive enhancement report"""
    print("\n=== Generating Progressive Enhancement Report ===")
    
    report = []
    report.append("# Progressive Enhancement Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # Dataset overview
    report.append("## Dataset Overview")
    report.append(f"Total videos analyzed: {len(df)}")
    report.append(f"Positive contact cases (manual): {y_true.sum()} ({y_true.sum()/len(y_true)*100:.1f}%)")
    report.append(f"Negative contact cases (manual): {len(y_true) - y_true.sum()} ({(len(y_true) - y_true.sum())/len(y_true)*100:.1f}%)")
    report.append("")
    
    # Progressive method performance
    report.append("## Progressive Enhancement Performance")
    report.append("")
    for _, row in results_df.iterrows():
        report.append(f"### {row['Method']}")
        report.append(f"- Accuracy: {row['Accuracy']:.3f}")
        report.append(f"- Precision: {row['Precision']:.3f}")
        report.append(f"- Recall: {row['Recall']:.3f}")
        report.append(f"- F1-Score: {row['F1-Score']:.3f}")
        report.append(f"- False Positives: {row['False Positives']}")
        report.append(f"- False Negatives: {row['False Negatives']}")
        report.append("")
    
    # Enhancement effects
    base_performance = results_df[results_df['Method'] == 'Base Segmentation'].iloc[0]
    report.append("## Enhancement Effects vs Base Method")
    report.append("")
    
    for _, row in results_df.iterrows():
        if row['Method'] != 'Base Segmentation':
            f1_change = row['F1-Score'] - base_performance['F1-Score']
            recall_change = row['Recall'] - base_performance['Recall']
            precision_change = row['Precision'] - base_performance['Precision']
            fp_change = row['False Positives'] - base_performance['False Positives']
            fn_change = row['False Negatives'] - base_performance['False Negatives']
            
            report.append(f"### {row['Method']}")
            report.append(f"- F1-Score change: {f1_change:+.3f} ({f1_change/base_performance['F1-Score']*100:+.1f}%)")
            report.append(f"- Recall change: {recall_change:+.3f}")
            report.append(f"- Precision change: {precision_change:+.3f}")
            report.append(f"- False Positives change: {fp_change:+}")
            report.append(f"- False Negatives change: {fn_change:+}")
            report.append("")
    
    # Key findings
    best_f1_method = results_df.loc[results_df['F1-Score'].idxmax()]
    best_precision_method = results_df.loc[results_df['Precision'].idxmax()]
    best_recall_method = results_df.loc[results_df['Recall'].idxmax()]
    
    report.append("## Key Findings")
    report.append("")
    report.append(f"1. Best F1-Score: {best_f1_method['Method']} ({best_f1_method['F1-Score']:.3f})")
    report.append(f"2. Best Precision: {best_precision_method['Method']} ({best_precision_method['Precision']:.3f})")
    report.append(f"3. Best Recall: {best_recall_method['Method']} ({best_recall_method['Recall']:.3f})")
    
    combined_method = results_df[results_df['Method'] == 'All Combined'].iloc[0]
    base_method = results_df[results_df['Method'] == 'Base Segmentation'].iloc[0]
    overall_improvement = combined_method['F1-Score'] - base_method['F1-Score']
    
    report.append(f"4. Overall enhancement effect: {overall_improvement:+.3f} F1-Score improvement")
    report.append(f"5. False negative reduction: {base_method['False Negatives'] - combined_method['False Negatives']} videos")
    report.append(f"6. False positive change: {combined_method['False Positives'] - base_method['False Positives']:+} videos")
    
    # Save report
    with open('/home/jinqiao/mhi/results/progressive_enhancement_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Progressive enhancement report saved to: /home/jinqiao/mhi/results/progressive_enhancement_report.txt")
    
    return report

def main():
    """Main analysis function"""
    print("Progressive Enhancement Analysis for Contact Detection")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create progressive detection methods
    methods, y_true = create_progressive_detection_methods(df)
    
    # Compute metrics for each method
    results_df = compute_progressive_metrics(methods, y_true)
    
    # Analyze enhancement effects
    analyze_enhancement_effects(results_df)
    
    # Analyze enhancement overlap
    analyze_enhancement_overlap(df, methods, y_true)
    
    # Create visualizations
    fig = create_progressive_visualizations(results_df, df, methods, y_true)
    
    # Generate comprehensive report
    report = generate_progressive_report(results_df, df, methods, y_true)
    
    # Save results
    results_df.to_csv('/home/jinqiao/mhi/results/progressive_enhancement_metrics.csv', index=False)
    
    print("\n=== Progressive Enhancement Analysis Complete ===")
    print("Generated files:")
    print("- /home/jinqiao/mhi/results/progressive_enhancement_analysis.png")
    print("- /home/jinqiao/mhi/results/progressive_enhancement_report.txt")
    print("- /home/jinqiao/mhi/results/progressive_enhancement_metrics.csv")

if __name__ == "__main__":
    main()
"""
Visualization Module
Handles creation of plots and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


def plot_results(df, cv_scores, results, output_path):
    """
    Create comprehensive visualizations
    
    Args:
        df: Original dataframe with ratings
        cv_scores: Cross-validation scores
        results: Test evaluation results
        output_path: Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("\n" + "=" * 60)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Rating Distribution in Dataset
    ax1 = plt.subplot(2, 3, 1)
    rating_counts = df['rating'].value_counts().sort_index()
    bars = ax1.bar(rating_counts.index, rating_counts.values, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Rating', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Rating Distribution in Dataset', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=sorted(results['y_test'].unique()),
               yticklabels=sorted(results['y_test'].unique()),
               ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel('Predicted Rating', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Rating', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 3. Cross-Validation Scores
    ax3 = plt.subplot(2, 3, 3)
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bars = ax3.bar(folds, cv_scores, color='lightgreen',
                  edgecolor='darkgreen', alpha=0.7)
    ax3.axhline(y=cv_scores.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
    ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Model Performance Comparison
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['CV Mean\nAccuracy', 'Test\nAccuracy', 'Test\nF1-Score']
    values = [cv_scores.mean(), results['accuracy'], results['f1_score']]
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax4.barh(metrics, values, color=colors_list, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(axis='x', alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, values):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 5. True vs Predicted Distribution
    ax5 = plt.subplot(2, 3, 5)
    all_ratings = sorted(results['y_test'].unique())
    pred_counts = pd.Series(results['y_pred']).value_counts()
    true_counts = results['y_test'].value_counts()
    
    pred_data = [pred_counts.get(r, 0) for r in all_ratings]
    true_data = [true_counts.get(r, 0) for r in all_ratings]
    
    x = np.arange(len(all_ratings))
    width = 0.35
    bars1 = ax5.bar(x - width/2, true_data, width,
                   label='True Labels', color='lightcoral', alpha=0.7)
    bars2 = ax5.bar(x + width/2, pred_data, width,
                   label='Predictions', color='lightskyblue', alpha=0.7)
    ax5.set_xlabel('Rating', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax5.set_title('True vs Predicted Rating Distribution', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{int(r)}' for r in all_ratings])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Per-Class Accuracy
    ax6 = plt.subplot(2, 3, 6)
    per_class_acc = []
    classes = sorted(results['y_test'].unique())
    for cls in classes:
        mask = results['y_test'] == cls
        class_acc = accuracy_score(results['y_test'][mask], results['y_pred'][mask])
        per_class_acc.append(class_acc)
    
    bars = ax6.bar([f'Rating {c}' for c in classes], per_class_acc,
                  color='plum', edgecolor='purple', alpha=0.7)
    ax6.set_xlabel('Rating Class', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax6.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    return fig

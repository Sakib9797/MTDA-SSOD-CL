"""
Generate comprehensive result visualizations for IEEE paper
Creates publication-quality figures
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

OUTPUT_DIR = Path('paper_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
DOMAIN_NAMES = ['Normal', 'Foggy', 'Rainy']

def plot_training_curves():
    """Plot training curves with curriculum stages"""
    # Load training history
    with open('runs/train/training_history.json', 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_map = [h['val_map'] for h in history]
    map_normal = [h['val_map_normal'] for h in history]
    map_foggy = [h['val_map_foggy'] for h in history]
    map_rainy = [h['val_map_rainy'] for h in history]
    stages = [h['stage'] for h in history]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: mAP progression with curriculum stages
    ax1 = axes[0]
    
    # Shade curriculum stages
    stage_colors = ['#e8f4f8', '#fff4e6', '#ffeef0']
    stage_labels = ['Stage 1: Easy\n(Normal Only)', 'Stage 2: Medium\n(Normal + Foggy)', 'Stage 3: Hard\n(All Domains)']
    
    stage_boundaries = [0]
    for i in range(len(epochs)-1):
        if stages[i] != stages[i+1]:
            stage_boundaries.append(i+1)
    stage_boundaries.append(len(epochs))
    
    for i in range(len(stage_boundaries)-1):
        start = stage_boundaries[i]
        end = stage_boundaries[i+1]
        ax1.axvspan(epochs[start], epochs[end-1], alpha=0.3, color=stage_colors[i], label=stage_labels[i])
    
    # Plot curves
    ax1.plot(epochs, val_map, 'o-', linewidth=2.5, markersize=4, label='Overall mAP', color='#2c3e50')
    ax1.plot(epochs, map_normal, 's--', linewidth=1.5, markersize=3, label='Normal', color='#3498db', alpha=0.8)
    ax1.plot(epochs, map_foggy, '^--', linewidth=1.5, markersize=3, label='Foggy', color='#2ecc71', alpha=0.8)
    ax1.plot(epochs, map_rainy, 'd--', linewidth=1.5, markersize=3, label='Rainy', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Training Progress with Curriculum Learning', fontweight='bold', fontsize=13)
    ax1.legend(loc='lower right', frameon=True, shadow=True, ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(epochs)])
    
    # Plot 2: Training loss
    ax2 = axes[1]
    ax2.plot(epochs, train_loss, 'o-', linewidth=2, markersize=3, color='#e67e22')
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
    ax2.set_title('Training Loss Curve', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(epochs)])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'training_curves.png', bbox_inches='tight')
    print(f"✓ Saved: training_curves.pdf")
    plt.close()

def plot_domain_performance():
    """Plot per-domain performance comparison"""
    df = pd.read_csv('runs/train/test_results_by_domain.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: mAP comparison
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Val_mAP'], width, label='Validation', 
                   color='#3498db', edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, df['Test_mAP'], width, label='Test', 
                   color='#e74c3c', edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Domain', fontweight='bold', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Domain-Specific Performance', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Domain'])
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Precision, Recall, F1
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax2.bar(x - width, df['Test_Precision'], width, label='Precision', 
                   color='#2ecc71', edgecolor='black', linewidth=0.8)
    bars2 = ax2.bar(x, df['Test_Recall'], width, label='Recall', 
                   color='#f39c12', edgecolor='black', linewidth=0.8)
    bars3 = ax2.bar(x + width, df['Test_F1'], width, label='F1-Score', 
                   color='#9b59b6', edgecolor='black', linewidth=0.8)
    
    ax2.set_xlabel('Domain', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax2.set_title('Evaluation Metrics by Domain', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Domain'])
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_performance.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'domain_performance.png', bbox_inches='tight')
    print(f"✓ Saved: domain_performance.pdf")
    plt.close()

def plot_class_performance():
    """Plot per-class performance"""
    df = pd.read_csv('runs/train/test_results_by_class.csv')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: mAP by class
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    bars = ax1.bar(range(len(df)), df['mAP'], color=colors, 
                  edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, df['mAP']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Object Class', fontweight='bold', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Per-Class mAP Performance', fontweight='bold', fontsize=13)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_map = df['mAP'].mean()
    ax1.axhline(y=mean_map, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_map:.2f}%')
    ax1.legend(frameon=True, shadow=True)
    
    # Plot 2: Precision, Recall, F1 by class
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax2.bar(x - width, df['Precision'], width, label='Precision', 
                   color='#2ecc71', edgecolor='black', linewidth=0.8, alpha=0.8)
    bars2 = ax2.bar(x, df['Recall'], width, label='Recall', 
                   color='#f39c12', edgecolor='black', linewidth=0.8, alpha=0.8)
    bars3 = ax2.bar(x + width, df['F1'], width, label='F1-Score', 
                   color='#9b59b6', edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax2.set_xlabel('Object Class', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax2.set_title('Precision, Recall, and F1-Score by Class', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_performance.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'class_performance.png', bbox_inches='tight')
    print(f"✓ Saved: class_performance.pdf")
    plt.close()

def plot_confusion_matrix():
    """Plot confusion matrix style performance heatmap"""
    # Load class results
    df_class = pd.read_csv('runs/train/test_results_by_class.csv')
    
    # Create metric matrix
    metrics = ['Precision', 'Recall', 'F1']
    data = df_class[metrics].values.T
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(df_class)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(df_class['Class'])
    ax.set_yticklabels(metrics)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(df_class)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax.set_title('Performance Metrics Heatmap by Class', fontweight='bold', fontsize=13, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', bbox_inches='tight')
    print(f"✓ Saved: metrics_heatmap.pdf")
    plt.close()

def plot_curriculum_impact():
    """Plot curriculum learning impact"""
    # Load training history
    with open('runs/train/training_history.json', 'r') as f:
        history = json.load(f)
    
    # Calculate per-stage statistics
    stage_stats = {0: [], 1: [], 2: []}
    for h in history:
        stage_stats[h['stage']].append(h['val_map'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Box plot of mAP by stage
    ax1 = axes[0]
    stage_names = ['Stage 1\n(Easy)', 'Stage 2\n(Medium)', 'Stage 3\n(Hard)']
    data_to_plot = [stage_stats[0], stage_stats[1], stage_stats[2]]
    
    bp = ax1.boxplot(data_to_plot, labels=stage_names, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax1.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax1.set_title('mAP Distribution Across Curriculum Stages', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    means = [np.mean(stage_stats[i]) for i in range(3)]
    ax1.plot(range(1, 4), means, 'D-', color='green', linewidth=2, markersize=8, label='Mean')
    ax1.legend(frameon=True, shadow=True)
    
    # Plot 2: Stage progression
    ax2 = axes[1]
    stage_labels = ['Stage 1', 'Stage 2', 'Stage 3']
    initial_maps = [stage_stats[i][0] for i in range(3)]
    final_maps = [stage_stats[i][-1] for i in range(3)]
    improvements = [final_maps[i] - initial_maps[i] for i in range(3)]
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, initial_maps, width, label='Initial mAP', 
                   color='#e74c3c', edgecolor='black', linewidth=0.8)
    bars2 = ax2.bar(x + width/2, final_maps, width, label='Final mAP', 
                   color='#2ecc71', edgecolor='black', linewidth=0.8)
    
    # Add improvement annotations
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
        ax2.annotate(f'+{imp:.1f}%',
                    xy=(i, max(bar1.get_height(), bar2.get_height())),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color='green')
    
    ax2.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Performance Improvement per Stage', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_labels)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'curriculum_impact.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'curriculum_impact.png', bbox_inches='tight')
    print(f"✓ Saved: curriculum_impact.pdf")
    plt.close()

def generate_results_tables():
    """Generate LaTeX tables for results"""
    # Domain results table
    df_domain = pd.read_csv('runs/train/test_results_by_domain.csv')
    latex_domain = df_domain.to_latex(index=False, float_format='%.2f',
                                     caption='Performance Comparison Across Domains',
                                     label='tab:domain_results',
                                     column_format='lcccccc')
    
    with open(OUTPUT_DIR / 'domain_results.tex', 'w') as f:
        f.write(latex_domain)
    print(f"✓ Saved: domain_results.tex")
    
    # Class results table
    df_class = pd.read_csv('runs/train/test_results_by_class.csv')
    latex_class = df_class.to_latex(index=False, float_format='%.2f',
                                   caption='Per-Class Performance Metrics',
                                   label='tab:class_results',
                                   column_format='lccccc')
    
    with open(OUTPUT_DIR / 'class_results.tex', 'w') as f:
        f.write(latex_class)
    print(f"✓ Saved: class_results.tex")

def main():
    print("="*60)
    print("Generating Results Visualizations for IEEE Paper")
    print("="*60)
    
    print("\n[1/6] Plotting training curves...")
    plot_training_curves()
    
    print("\n[2/6] Plotting domain performance...")
    plot_domain_performance()
    
    print("\n[3/6] Plotting class performance...")
    plot_class_performance()
    
    print("\n[4/6] Plotting metrics heatmap...")
    plot_confusion_matrix()
    
    print("\n[5/6] Plotting curriculum impact...")
    plot_curriculum_impact()
    
    print("\n[6/6] Generating LaTeX tables...")
    generate_results_tables()
    
    print("\n" + "="*60)
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == '__main__':
    main()

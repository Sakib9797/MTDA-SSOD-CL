"""
Exploratory Data Analysis (EDA) for Multi-Domain Object Detection Dataset
Generates comprehensive visualizations for paper
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Dataset paths
DATASET_ROOT = Path('dataset')
OUTPUT_DIR = Path('paper_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Class names
CLASS_NAMES = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
DOMAIN_NAMES = ['normal', 'foggy', 'rainy']

def analyze_dataset_statistics():
    """Analyze dataset size, distribution across domains and splits"""
    stats = defaultdict(lambda: defaultdict(int))
    
    for domain in DOMAIN_NAMES:
        for split in ['train', 'val']:
            img_dir = DATASET_ROOT / domain / split / 'images'
            label_dir = DATASET_ROOT / domain / split / 'labels'
            
            if img_dir.exists():
                images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                labels = list(label_dir.glob('*.txt'))
                
                stats[domain][f'{split}_images'] = len(images)
                stats[domain][f'{split}_labels'] = len(labels)
    
    return stats

def count_objects_per_class():
    """Count objects per class across domains"""
    class_counts = defaultdict(lambda: defaultdict(int))
    
    for domain in DOMAIN_NAMES:
        for split in ['train', 'val']:
            label_dir = DATASET_ROOT / domain / split / 'labels'
            
            if label_dir.exists():
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(CLASS_NAMES):
                                    class_counts[domain][class_id] += 1
                                    class_counts['total'][class_id] += 1
    
    return class_counts

def analyze_bounding_box_statistics():
    """Analyze bounding box sizes and aspect ratios"""
    box_stats = defaultdict(lambda: {'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': []})
    
    for domain in DOMAIN_NAMES:
        for split in ['train', 'val']:
            label_dir = DATASET_ROOT / domain / split / 'labels'
            
            if label_dir.exists():
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                box_stats[domain]['widths'].append(w)
                                box_stats[domain]['heights'].append(h)
                                box_stats[domain]['areas'].append(w * h)
                                if h > 0:
                                    box_stats[domain]['aspect_ratios'].append(w / h)
    
    return box_stats

def plot_dataset_distribution():
    """Plot 1: Dataset distribution across domains and splits"""
    stats = analyze_dataset_statistics()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Prepare data
    domains = list(stats.keys())
    train_counts = [stats[d]['train_images'] for d in domains]
    val_counts = [stats[d]['val_images'] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, val_counts, width, label='Validation', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Domain', fontweight='bold')
    ax.set_ylabel('Number of Images', fontweight='bold')
    ax.set_title('Dataset Distribution Across Domains', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'dataset_distribution.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'dataset_distribution.pdf'}")
    plt.close()

def plot_class_distribution():
    """Plot 2: Class distribution across domains"""
    class_counts = count_objects_per_class()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot for each domain + total
    plot_domains = DOMAIN_NAMES + ['total']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, domain in enumerate(plot_domains):
        ax = axes[idx]
        
        counts = [class_counts[domain][i] for i in range(len(CLASS_NAMES))]
        
        bars = ax.bar(range(len(CLASS_NAMES)), counts, color=colors[idx], 
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Object Class', fontweight='bold')
        ax.set_ylabel('Number of Instances', fontweight='bold')
        ax.set_title(f'Class Distribution - {domain.capitalize()}', fontweight='bold')
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'class_distribution.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'class_distribution.pdf'}")
    plt.close()

def plot_box_statistics():
    """Plot 3: Bounding box size and aspect ratio distributions"""
    box_stats = analyze_bounding_box_statistics()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot box areas
    for idx, domain in enumerate(DOMAIN_NAMES):
        ax = axes[0, idx]
        areas = box_stats[domain]['areas']
        
        ax.hist(areas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Normalized Area', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{domain.capitalize()} - Box Area', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_area = np.mean(areas)
        ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.3f}')
        ax.legend()
    
    # Plot aspect ratios
    for idx, domain in enumerate(DOMAIN_NAMES):
        ax = axes[1, idx]
        aspect_ratios = [ar for ar in box_stats[domain]['aspect_ratios'] if ar < 5]  # Filter outliers
        
        ax.hist(aspect_ratios, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Aspect Ratio (W/H)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{domain.capitalize()} - Aspect Ratio', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_ar = np.mean(aspect_ratios)
        ax.axvline(mean_ar, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ar:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'box_statistics.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'box_statistics.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'box_statistics.pdf'}")
    plt.close()

def plot_class_imbalance():
    """Plot 4: Class imbalance ratio visualization"""
    class_counts = count_objects_per_class()
    
    total_counts = [class_counts['total'][i] for i in range(len(CLASS_NAMES))]
    max_count = max(total_counts)
    imbalance_ratios = [max_count / count if count > 0 else 0 for count in total_counts]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['#2ecc71' if ratio < 2 else '#f39c12' if ratio < 4 else '#e74c3c' for ratio in imbalance_ratios]
    bars = ax.bar(range(len(CLASS_NAMES)), imbalance_ratios, color=colors, 
                 edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add value labels
    for bar, ratio in zip(bars, imbalance_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.2f}x',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Balanced (1x)')
    ax.set_xlabel('Object Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Imbalance Ratio (max/current)', fontweight='bold', fontsize=12)
    ax.set_title('Class Imbalance Analysis', fontweight='bold', fontsize=14)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_imbalance.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'class_imbalance.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'class_imbalance.pdf'}")
    plt.close()

def generate_summary_table():
    """Generate LaTeX table with dataset statistics"""
    stats = analyze_dataset_statistics()
    class_counts = count_objects_per_class()
    
    # Create DataFrame
    data = []
    for domain in DOMAIN_NAMES:
        train_imgs = stats[domain]['train_images']
        val_imgs = stats[domain]['val_images']
        total_imgs = train_imgs + val_imgs
        
        train_objs = sum(class_counts[domain].values()) if domain in class_counts else 0
        
        data.append({
            'Domain': domain.capitalize(),
            'Train Images': train_imgs,
            'Val Images': val_imgs,
            'Total Images': total_imgs,
            'Total Objects': train_objs
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(OUTPUT_DIR / 'dataset_statistics.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, caption='Dataset Statistics Across Domains', 
                             label='tab:dataset_stats', column_format='lcccc')
    
    with open(OUTPUT_DIR / 'dataset_statistics.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Saved: {OUTPUT_DIR / 'dataset_statistics.csv'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'dataset_statistics.tex'}")
    
    return df

def main():
    print("="*60)
    print("Generating EDA Visualizations for IEEE Paper")
    print("="*60)
    
    print("\n[1/5] Plotting dataset distribution...")
    plot_dataset_distribution()
    
    print("\n[2/5] Plotting class distribution...")
    plot_class_distribution()
    
    print("\n[3/5] Plotting bounding box statistics...")
    plot_box_statistics()
    
    print("\n[4/5] Plotting class imbalance...")
    plot_class_imbalance()
    
    print("\n[5/5] Generating summary table...")
    df = generate_summary_table()
    
    print("\n" + "="*60)
    print("EDA Complete! Summary:")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == '__main__':
    main()

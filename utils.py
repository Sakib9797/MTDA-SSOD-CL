"""
Utility functions for Multi-Domain Object Detection
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def plot_training_history(history_path, output_dir='runs/train'):
    """Plot training metrics"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_map = [h['val_map'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # mAP plot
    axes[1].plot(epochs, val_map, 'r-', label='Validation mAP')
    axes[1].axhline(y=40, color='g', linestyle='--', label='Target mAP (40)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('Validation mAP')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png', dpi=150)
    print(f"Training curves saved to {output_dir}/training_curves.png")

def plot_domain_comparison(history_path, output_dir='runs/train'):
    """Plot per-domain mAP comparison"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    map_normal = [h['val_map_normal'] for h in history]
    map_foggy = [h['val_map_foggy'] for h in history]
    map_rainy = [h['val_map_rainy'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, map_normal, 'g-', label='Normal', linewidth=2)
    plt.plot(epochs, map_foggy, 'b-', label='Foggy', linewidth=2)
    plt.plot(epochs, map_rainy, 'r-', label='Rainy', linewidth=2)
    plt.axhline(y=40, color='k', linestyle='--', alpha=0.5, label='Target (40)')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.title('Per-Domain Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'domain_comparison.png', dpi=150)
    print(f"Domain comparison saved to {output_dir}/domain_comparison.png")

def plot_curriculum_stages(history_path, output_dir='runs/train'):
    """Plot curriculum learning stages"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    stages = [h['stage'] for h in history]
    val_map = [h['val_map'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot mAP
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mAP', color=color, fontsize=12)
    ax1.plot(epochs, val_map, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot curriculum stage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Curriculum Stage', color=color, fontsize=12)
    ax2.plot(epochs, stages, color=color, linewidth=2, linestyle='--', marker='o', markersize=3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Easy', 'Medium', 'Hard'])
    
    plt.title('Curriculum Learning Progress', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'curriculum_stages.png', dpi=150)
    print(f"Curriculum stages saved to {output_dir}/curriculum_stages.png")

def visualize_all_results(output_dir='runs/train'):
    """Generate all visualization plots"""
    history_path = Path(output_dir) / 'training_history.json'
    
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return
    
    print("Generating visualization plots...")
    plot_training_history(history_path, output_dir)
    plot_domain_comparison(history_path, output_dir)
    plot_curriculum_stages(history_path, output_dir)
    print("All plots generated successfully!")

def compute_flops(model, img_size=640):
    """Compute model FLOPs"""
    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(input_tensor,))
        return flops, params
    except:
        return None, None

def print_model_summary(model):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")

if __name__ == '__main__':
    # Generate visualizations
    visualize_all_results()

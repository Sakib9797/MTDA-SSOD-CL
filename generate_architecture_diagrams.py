"""
Generate comprehensive architecture and flow diagrams for the paper
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Generating Architecture and Flow Diagrams")
print("=" * 70)

# 1. Overall System Architecture
print("\n[1/3] Creating overall system architecture...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Multi-Domain Object Detection Architecture', 
        ha='center', fontsize=14, fontweight='bold')

# Input Layer
input_box = FancyBboxPatch((0.5, 8), 9, 0.8, boxstyle="round,pad=0.1", 
                           edgecolor='#2c3e50', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.4, 'Input: Multi-Domain Images (Normal, Foggy, Rainy)', 
        ha='center', fontsize=10, fontweight='bold')

# Arrow
ax.arrow(5, 7.9, 0, -0.5, head_width=0.2, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)

# Curriculum Learning Stage
stage_box = FancyBboxPatch((0.5, 6.2), 9, 1.4, boxstyle="round,pad=0.1",
                          edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(stage_box)
ax.text(5, 7.3, 'Curriculum Learning Strategy', ha='center', fontsize=11, fontweight='bold', color='#c0392b')
ax.text(1.8, 6.8, 'Stage 1: Easy\n(Epochs 1-20)\nNormal', ha='center', fontsize=8)
ax.text(5, 6.8, 'Stage 2: Medium\n(Epochs 21-60)\nNormal + Foggy', ha='center', fontsize=8)
ax.text(8.2, 6.8, 'Stage 3: Hard\n(Epochs 61-100)\nAll Domains', ha='center', fontsize=8)

# Arrow
ax.arrow(5, 6.1, 0, -0.5, head_width=0.2, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)

# YOLOv11 Backbone
backbone_box = FancyBboxPatch((0.5, 4.5), 9, 1.3, boxstyle="round,pad=0.1",
                             edgecolor='#3498db', facecolor='#d6eaf8', linewidth=2)
ax.add_patch(backbone_box)
ax.text(5, 5.5, 'YOLOv11s Backbone (9.4M params)', ha='center', fontsize=11, fontweight='bold', color='#21618c')
ax.text(5, 5.1, 'CSPDarknet53 + PANet + SPPF', ha='center', fontsize=9)
ax.text(5, 4.8, 'Feature Maps: [20×20, 40×40, 80×80]', ha='center', fontsize=8, style='italic')

# Arrow
ax.arrow(5, 4.4, 0, -0.5, head_width=0.2, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)

# Domain Adaptation Module
da_box = FancyBboxPatch((0.5, 2.8), 4, 1.3, boxstyle="round,pad=0.1",
                        edgecolor='#9b59b6', facecolor='#ebdef0', linewidth=2)
ax.add_patch(da_box)
ax.text(2.5, 3.6, 'Domain Adaptation', ha='center', fontsize=10, fontweight='bold', color='#6c3483')
ax.text(2.5, 3.3, '• Adversarial Training', ha='center', fontsize=8)
ax.text(2.5, 3.05, '• Feature Alignment', ha='center', fontsize=8)

# Detection Head
det_box = FancyBboxPatch((5.5, 2.8), 4, 1.3, boxstyle="round,pad=0.1",
                        edgecolor='#16a085', facecolor='#d1f2eb', linewidth=2)
ax.add_patch(det_box)
ax.text(7.5, 3.6, 'Detection Head', ha='center', fontsize=10, fontweight='bold', color='#0e6655')
ax.text(7.5, 3.3, '• Multi-scale Predictions', ha='center', fontsize=8)
ax.text(7.5, 3.05, '• NMS Post-processing', ha='center', fontsize=8)

# Arrows to output
ax.arrow(2.5, 2.7, 0, -0.5, head_width=0.15, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)
ax.arrow(7.5, 2.7, 0, -0.5, head_width=0.15, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)

# Output Layer
output_box = FancyBboxPatch((0.5, 1), 9, 0.9, boxstyle="round,pad=0.1",
                           edgecolor='#27ae60', facecolor='#d5f4e6', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.5, 'Output: Object Detections (7 classes) + Confidence Scores', 
        ha='center', fontsize=10, fontweight='bold', color='#186a3b')

# Performance Metrics
metrics_box = FancyBboxPatch((0.5, 0.1), 9, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='#f39c12', facecolor='#fef5e7', linewidth=2)
ax.add_patch(metrics_box)
ax.text(5, 0.55, 'Final Performance: mAP = 53.02%', ha='center', fontsize=10, fontweight='bold', color='#d68910')
ax.text(5, 0.3, 'Normal: 56.98%  |  Foggy: 54.56%  |  Rainy: 47.52%', 
        ha='center', fontsize=9, color='#7d6608')

plt.tight_layout()
plt.savefig(output_dir / 'system_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: system_architecture.pdf/png")

# 2. Training Pipeline Flow
print("\n[2/3] Creating training pipeline flow...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7, 5.5, 'Training Pipeline Workflow', ha='center', fontsize=14, fontweight='bold')

# Step boxes
steps = [
    ('Data\nLoading', 1, 4, '#3498db'),
    ('Curriculum\nScheduler', 3, 4, '#9b59b6'),
    ('Data\nAugmentation', 5, 4, '#e67e22'),
    ('Forward\nPass', 7, 4, '#16a085'),
    ('Loss\nComputation', 9, 4, '#e74c3c'),
    ('Backward\nPass', 11, 4, '#2ecc71'),
    ('Validation', 13, 4, '#f39c12')
]

for i, (text, x, y, color) in enumerate(steps):
    box = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1, boxstyle="round,pad=0.08",
                         edgecolor=color, facecolor=f'{color}33', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows between steps
    if i < len(steps) - 1:
        ax.arrow(x+0.7, y, 0.5, 0, head_width=0.15, head_length=0.15, 
                fc='#2c3e50', ec='#2c3e50', lw=1.5)

# Add epoch loop indicator
loop_arrow = FancyArrowPatch((13, 3.3), (1, 3.3), arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='#c0392b', linestyle='--')
ax.add_patch(loop_arrow)
ax.text(7, 3, 'Repeat for 100 Epochs', ha='center', fontsize=10, 
        fontweight='bold', color='#c0392b', style='italic')

# Add detailed info boxes below
info_boxes = [
    ('Multi-Domain\nDataset\n1,880 images', 1, 1.5),
    ('3-Stage\nCurriculum\nDynamic', 3, 1.5),
    ('Flip, Scale\nHSV, Mosaic\nAugmentations', 5, 1.5),
    ('YOLOv11s\nBackbone\n9.4M params', 7, 1.5),
    ('Multi-task:\nBox + Cls\n+ Domain', 9, 1.5),
    ('Adam\nOptimizer\nLR=0.001', 11, 1.5),
    ('Per-Domain\nmAP\nMetrics', 13, 1.5)
]

for text, x, y in info_boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=7, 
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', edgecolor='#7f8c8d', linewidth=1))

plt.tight_layout()
plt.savefig(output_dir / 'training_pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'training_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: training_pipeline.pdf/png")

# 3. Model Architecture Details
print("\n[3/3] Creating detailed model architecture...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'YOLOv11s Model Architecture', ha='center', fontsize=14, fontweight='bold')

# Layer structure
layers = [
    ('Input Image\n640×640×3', 5, 8.5, '#3498db', 1.5),
    ('Conv + BN + SiLU\n320×320×32', 5, 7.5, '#9b59b6', 1.2),
    ('C2f Block 1\n160×160×64', 5, 6.6, '#e67e22', 1.2),
    ('C2f Block 2\n80×80×128', 5, 5.7, '#e67e22', 1.2),
    ('C2f Block 3\n40×40×256', 5, 4.8, '#e67e22', 1.2),
    ('SPPF Layer\n40×40×512', 5, 3.9, '#16a085', 1.2),
    ('PANet Head\nMulti-scale Features', 5, 3.0, '#e74c3c', 1.2),
    ('Detection Layers\n[20×20, 40×40, 80×80]', 5, 2.1, '#2ecc71', 1.5),
    ('Output\nBBoxes + Classes', 5, 0.9, '#f39c12', 1.2),
]

for i, (text, x, y, color, height) in enumerate(layers):
    box = FancyBboxPatch((x-1.2, y-height/2), 2.4, height, boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor=f'{color}33', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add arrows between layers
    if i < len(layers) - 1:
        next_y = layers[i+1][2]
        arrow_start_y = y - height/2 - 0.1
        arrow_end_y = next_y + layers[i+1][4]/2 + 0.1
        ax.arrow(x, arrow_start_y, 0, arrow_end_y - arrow_start_y, 
                head_width=0.15, head_length=0.1, fc='#34495e', ec='#34495e', lw=2)

# Add side annotations
annotations = [
    ('Backbone\nFeature\nExtraction', 7.5, 6, '#2c3e50'),
    ('Neck\nFeature\nFusion', 7.5, 3.5, '#2c3e50'),
    ('Head\nPrediction', 7.5, 1.5, '#2c3e50'),
]

for text, x, y, color in annotations:
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', 
           color=color, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', 
           edgecolor=color, linewidth=2))

# Add parameter count
param_text = 'Total Parameters: 9.4M\nFLOPs: 16.3 GFLOPs\nSpeed: ~140 FPS (RTX 4060)'
ax.text(5, 0.2, param_text, ha='center', fontsize=8, 
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef5e7', edgecolor='#f39c12', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / 'model_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: model_architecture.pdf/png")

print("\n" + "=" * 70)
print("✓ All architecture diagrams generated successfully!")
print("=" * 70)
print(f"\nSaved to: {output_dir}/")
print("  • system_architecture.pdf/png")
print("  • training_pipeline.pdf/png")
print("  • model_architecture.pdf/png")
print("=" * 70)

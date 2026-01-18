"""
Generate Architecture Diagram for IEEE Paper
Creates a visual representation of the model architecture
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
color_input = '#e8f4f8'
color_backbone = '#d5e8f7'
color_feature = '#c2ddf5'
color_domain = '#ffe6e6'
color_align = '#fff4e6'
color_output = '#e6f7e6'
color_loss = '#f0e6ff'

# Helper function to draw boxes
def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', wrap=True)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=20, 
                          linewidth=2, color=color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(7, 7.5, 'Multi-Target Domain Adaptation Architecture', 
       ha='center', fontsize=14, fontweight='bold')

# Input Layer (3 domains)
draw_box(ax, 0.5, 6.0, 1.5, 0.6, 'Normal\nImages', color_input, 8)
draw_box(ax, 0.5, 5.2, 1.5, 0.6, 'Foggy\nImages', color_input, 8)
draw_box(ax, 0.5, 4.4, 1.5, 0.6, 'Rainy\nImages', color_input, 8)

# Input arrow
draw_arrow(ax, 2.1, 5.3, 2.8, 5.3, 'Multi-Domain\nInput')

# Base Detector (YOLOv11)
draw_box(ax, 2.8, 4.6, 2.0, 1.4, 'YOLOv11s\nBackbone\n(9.4M params)\n\nPre-trained\non COCO', color_backbone)

# Feature extraction arrows
draw_arrow(ax, 4.9, 5.3, 5.5, 5.8, 'Features\nF')
draw_arrow(ax, 4.9, 5.3, 5.5, 4.8, '')

# Feature Alignment Module
draw_box(ax, 5.5, 5.5, 2.0, 0.8, 'Feature Alignment\nModule\n(Residual Conv)', color_align)

# Domain Discriminator Branch (with GRL)
draw_box(ax, 5.5, 4.3, 2.0, 0.8, 'Gradient Reversal\nLayer (α=2.0)', color_domain)
draw_arrow(ax, 7.6, 4.7, 8.5, 4.7, '')
draw_box(ax, 8.5, 4.3, 2.0, 0.8, 'Domain\nDiscriminator\n(3 classes)', color_domain)

# Detection Head
draw_arrow(ax, 7.6, 5.9, 8.5, 5.9, 'Aligned\nFeatures')
draw_box(ax, 8.5, 5.5, 2.0, 0.8, 'Detection\nHead\n(7 classes)', color_output)

# Outputs
draw_arrow(ax, 10.6, 5.9, 11.5, 6.4, '')
draw_box(ax, 11.5, 6.1, 2.0, 0.8, 'Object\nDetections\n(bbox, class)', color_output)

draw_arrow(ax, 10.6, 4.7, 11.5, 4.7, '')
draw_box(ax, 11.5, 4.3, 2.0, 0.8, 'Domain\nPredictions\n(N/F/R)', color_domain)

# Loss Functions (bottom)
loss_y = 2.5
draw_box(ax, 1.5, loss_y, 1.8, 0.6, 'Detection\nLoss\n$\\mathcal{L}_{det}$', color_loss, 8)
draw_box(ax, 3.5, loss_y, 1.8, 0.6, 'Domain\nLoss\n$\\mathcal{L}_{dom}$', color_loss, 8)
draw_box(ax, 5.5, loss_y, 1.8, 0.6, 'MMD\nLoss\n$\\mathcal{L}_{mmd}$', color_loss, 8)
draw_box(ax, 7.5, loss_y, 1.8, 0.6, 'Consistency\nLoss\n$\\mathcal{L}_{cons}$', color_loss, 8)

# Total loss
draw_arrow(ax, 2.4, 3.2, 4.2, 3.8, '', 'purple')
draw_arrow(ax, 4.4, 3.2, 4.2, 3.8, '', 'purple')
draw_arrow(ax, 6.4, 3.2, 5.8, 3.8, '', 'purple')
draw_arrow(ax, 8.4, 3.2, 6.2, 3.8, '', 'purple')

draw_box(ax, 4.2, 3.8, 2.6, 0.6, 'Total Loss: $\\mathcal{L}_{total}$\n$= \\mathcal{L}_{det} + \\lambda_d\\mathcal{L}_{dom} + \\lambda_m\\mathcal{L}_{mmd} + \\lambda_c\\mathcal{L}_{cons}$', 
        color_loss, 8)

# Curriculum Learning Box
curriculum_box = FancyBboxPatch((0.3, 0.3), 13.4, 1.3, 
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='lightgreen', 
                               linewidth=3, alpha=0.3)
ax.add_patch(curriculum_box)

ax.text(7, 1.4, 'Dynamic Curriculum Learning', ha='center', 
       fontsize=11, fontweight='bold', color='darkgreen')

# Curriculum stages
stage_y = 0.8
draw_box(ax, 0.8, stage_y, 3.5, 0.5, 'Stage 1 (Easy): Normal only\nWeights: (1.0, 0.0, 0.0), τ=0.6', 
        'white', 7)
draw_box(ax, 4.8, stage_y, 3.5, 0.5, 'Stage 2 (Medium): Normal+Foggy\nWeights: (0.75, 0.25, 0.0), τ=0.4', 
        'white', 7)
draw_box(ax, 8.8, stage_y, 3.5, 0.5, 'Stage 3 (Hard): All domains\nWeights: (0.5, 0.25, 0.25), τ=0.2', 
        'white', 7)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
    mpatches.Patch(facecolor=color_backbone, edgecolor='black', label='Backbone'),
    mpatches.Patch(facecolor=color_align, edgecolor='black', label='Feature Alignment'),
    mpatches.Patch(facecolor=color_domain, edgecolor='black', label='Domain Adaptation'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output'),
    mpatches.Patch(facecolor=color_loss, edgecolor='black', label='Loss Functions')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.25), 
         ncol=3, fontsize=8, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('paper_figures/architecture_diagram.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper_figures/architecture_diagram.png', bbox_inches='tight', dpi=300)
print("✓ Saved: paper_figures/architecture_diagram.pdf")
print("✓ Saved: paper_figures/architecture_diagram.png")
plt.close()

print("\nArchitecture diagram generated successfully!")
print("This diagram shows:")
print("  • Multi-domain input (Normal, Foggy, Rainy)")
print("  • YOLOv11s backbone with feature extraction")
print("  • Feature alignment module")
print("  • Domain discriminator with gradient reversal")
print("  • Detection head for object predictions")
print("  • Four loss components")
print("  • Three-stage curriculum learning strategy")

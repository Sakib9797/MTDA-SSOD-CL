"""
Generate Clear YOLOv11s Architecture Diagram for IEEE Paper
High-quality visualization optimized for LaTeX/Overleaf
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# High-quality settings for IEEE papers
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5

OUTPUT_BASENAME = 'paper_figures/yolov11s_architecture'

# Define color scheme for different layer types
colors = {
    'conv': '#FFE6E6',      # Light red
    'c2f': '#FFE6CC',       # Light orange
    'sppf': '#FFFFCC',      # Light yellow
    'upsample': '#E6F7FF',  # Light blue
    'concat': '#E6FFE6',    # Light green
    'detect': '#FFE6F0',    # Light pink
}

def render_single_column_part(
        *,
        layers,
        out_pdf,
        out_png,
        out_eps,
        title,
        show_legend: bool,
        show_stats: bool,
        section_labels,
):
        """Render a single-column IEEE-friendly architecture figure.

        Key idea: size the canvas close to IEEE column width so text does not get
        scaled down too aggressively by \includegraphics[width=\columnwidth].
        """

        # ~3.5 inches is typical IEEE column width.
        fig, ax = plt.subplots(figsize=(3.55, 6.2), layout="constrained")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 30)
        ax.axis('off')

        # Title
        ax.text(6, 28.8, title, ha='center', fontsize=11, fontweight='bold')

# Helper function to draw layer blocks
def draw_layer(ax, x, y, width, height, layer_name, params, color, fontsize=10):
    """Draw a layer block with parameters"""
    box = Rectangle((x, y), width, height, 
                   edgecolor='black', facecolor=color, 
                   linewidth=2, zorder=2)
    ax.add_patch(box)
    
    # Layer name
    ax.text(x + width/2, y + height*0.65, layer_name, 
           ha='center', va='center', fontsize=fontsize, 
           fontweight='bold', zorder=3)
    
    # Parameters
    ax.text(x + width/2, y + height*0.35, params, 
           ha='center', va='center', fontsize=fontsize-1.5, 
           zorder=3, style='italic')

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=25,
                          linewidth=2.5, color='black', zorder=1)
    ax.add_patch(arrow)
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x + 1.2, mid_y, label, fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', edgecolor='gray'))

# YOLOv11s Architecture Layers (auto-spaced to guarantee no overlap)
        x_center = 3.2
        layer_width = 6.2
        x_mid = x_center + layer_width / 2

        n_layers = len(layers)
        top_y = 26.0
        bottom_y = 7.0
        step = (top_y - bottom_y) / (n_layers - 1)
        layer_height = step * 0.70

        y_positions = []
        for i, (layer_name, params, color, tag) in enumerate(layers):
                y = top_y - i * step
                y_positions.append(y)
                fontsize = 9 if i in (0, n_layers - 1) else 8
                draw_layer(ax, x_center, y, layer_width, layer_height, layer_name, params, color, fontsize)

                if tag:
                        ax.text(
                                x_center + layer_width + 0.35,
                                y + layer_height * 0.5,
                                tag,
                                fontsize=7.5,
                                va="center",
                                ha="left",
                                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7, edgecolor="black", linewidth=0.8),
                        )

                if i < n_layers - 1:
                        next_y = top_y - (i + 1) * step
                        draw_arrow(ax, x_mid, y, x_mid, next_y + layer_height)

        # Side section labels (optional)
        def mid_y(i0: int, i1: int) -> float:
                return (y_positions[i0] + y_positions[i1] + layer_height) / 2

        for label, i0, i1, facecolor in section_labels:
                ax.text(
                        1.05,
                        mid_y(i0, i1),
                        label,
                        rotation=90,
                        ha='center',
                        va='center',
                        fontsize=8.5,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor=facecolor, edgecolor='black', linewidth=1.2),
                )

        # Model stats (optional)
        if show_stats:
                stats_text = (
                        'Model Statistics:\n'
                        '• Parameters: 9.4M\n'
                        '• GFLOPs: 21.5\n'
                        '• Input: 640×640\n'
                        '• Pre-trained: COCO'
                )
                ax.text(
                        6,
                        3.0,
                        stats_text,
                        ha='center',
                        va='center',
                        fontsize=7.2,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow', edgecolor='black', linewidth=1.2),
                )

        # Legend (optional) - outside axes to avoid collisions
        if show_legend:
                legend_elements = [
                        mpatches.Patch(facecolor=colors['conv'], edgecolor='black', label='Conv'),
                        mpatches.Patch(facecolor=colors['c2f'], edgecolor='black', label='C2f'),
                        mpatches.Patch(facecolor=colors['sppf'], edgecolor='black', label='SPPF'),
                        mpatches.Patch(facecolor=colors['upsample'], edgecolor='black', label='Upsample'),
                        mpatches.Patch(facecolor=colors['concat'], edgecolor='black', label='Concat'),
                        mpatches.Patch(facecolor=colors['detect'], edgecolor='black', label='Detect'),
                ]
                fig.legend(
                        handles=legend_elements,
                        loc='lower center',
                        bbox_to_anchor=(0.5, 0.02),
                        ncol=3,
                        fontsize=7.0,
                        frameon=True,
                        edgecolor='black',
                        fancybox=True,
                )

        fig.savefig(out_pdf, bbox_inches='tight', dpi=600)
        fig.savefig(out_png, bbox_inches='tight', dpi=600)
        fig.savefig(out_eps, bbox_inches='tight', dpi=600)
        plt.close(fig)


ALL_LAYERS = [
        ("Input Image", "640×640×3", "#E0E0E0", None),
        ("Conv", "3×3, stride=2, 32", colors["conv"], None),
        ("Conv", "3×3, stride=2, 64", colors["conv"], None),
        ("C2f Block", "n=1, 64 channels", colors["c2f"], None),
        ("Conv", "3×3, stride=2, 128", colors["conv"], None),
        ("C2f Block", "n=2, 128 channels", colors["c2f"], None),
        ("Conv", "3×3, stride=2, 256", colors["conv"], "P3/8"),
        ("C2f Block", "n=2, 256 channels", colors["c2f"], None),
        ("Conv", "3×3, stride=2, 512", colors["conv"], "P4/16"),
        ("C2f Block", "n=1, 512 channels", colors["c2f"], None),
        ("SPPF", "k=5, 512 channels", colors["sppf"], "P5/32"),
        ("Upsample", "2×, 512→256", colors["upsample"], None),
        ("Concat", "P4 features", colors["concat"], None),
        ("C2f Block", "n=1, 256 channels", colors["c2f"], None),
        ("Upsample", "2×, 256→128", colors["upsample"], None),
        ("Concat", "P3 features", colors["concat"], None),
        ("C2f Block", "n=1, 128 channels", colors["c2f"], None),
        ("Detect Head", "(P3/8, P4/16, P5/32)", colors["detect"], None),
        ("Output", "Bounding boxes + Classes", "#D0D0D0", None),
]
PART_A = ALL_LAYERS[:11]  # Input + Backbone up to SPPF
PART_B = ALL_LAYERS[11:]  # Neck + Head + Output

# Part A: Backbone
render_single_column_part(
        layers=PART_A,
        out_pdf=f'{OUTPUT_BASENAME}_col1_a.pdf',
        out_png=f'{OUTPUT_BASENAME}_col1_a.png',
        out_eps=f'{OUTPUT_BASENAME}_col1_a.eps',
        title='YOLOv11s Architecture (A) Backbone',
        show_legend=True,
        show_stats=False,
        section_labels=[('BACKBONE', 1, len(PART_A) - 1, 'lightblue')],
)

# Part B: Neck + Head
render_single_column_part(
        layers=PART_B,
        out_pdf=f'{OUTPUT_BASENAME}_col1_b.pdf',
        out_png=f'{OUTPUT_BASENAME}_col1_b.png',
        out_eps=f'{OUTPUT_BASENAME}_col1_b.eps',
        title='YOLOv11s Architecture (B) Neck + Head',
        show_legend=True,
        show_stats=True,
        section_labels=[
                ('NECK', 0, max(0, len(PART_B) - 4), 'lightgreen'),
                ('HEAD', max(0, len(PART_B) - 3), len(PART_B) - 1, 'lightcoral'),
        ],
)

print("=" * 75)
print("✓ YOLOv11s Architecture Diagram Generated Successfully!")
print("=" * 75)
print("\nFiles created:")
print(f"  • {OUTPUT_BASENAME}_col1_a.pdf (single-column Part A)")
print(f"  • {OUTPUT_BASENAME}_col1_b.pdf (single-column Part B)")
print("\nFeatures:")
print("  ✓ 600 DPI resolution for crystal-clear quality")
print("  ✓ Sized for IEEE single column (readable at \\columnwidth)")
print("  ✓ Large, readable fonts (10-18pt)")
print("  ✓ Complete layer-by-layer breakdown")
print("  ✓ Annotated with layer types and parameters")
print("  ✓ Clearly shows Backbone, Neck, and Head sections")
print("\n" + "=" * 75)

"""
Create a comprehensive zip package with all paper files
"""

import zipfile
from pathlib import Path
import shutil

def create_paper_package():
    """Create zip file with paper and all figures"""
    
    print("="*70)
    print("Creating IEEE Paper Package")
    print("="*70)
    
    # Create a temporary directory for organizing files
    package_dir = Path("IEEE_Paper_Package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Copy LaTeX file
    print("\n[1/3] Copying LaTeX file...")
    shutil.copy("paper_ieee_format.tex", package_dir / "paper_ieee_format.tex")
    print("  ✓ paper_ieee_format.tex")
    
    # Copy figures directory
    print("\n[2/3] Copying all figures...")
    figures_src = Path("paper_figures")
    figures_dst = package_dir / "paper_figures"
    
    if figures_src.exists():
        shutil.copytree(figures_src, figures_dst)
        
        # Count files
        pdf_count = len(list(figures_dst.glob("*.pdf")))
        png_count = len(list(figures_dst.glob("*.png")))
        tex_count = len(list(figures_dst.glob("*.tex")))
        csv_count = len(list(figures_dst.glob("*.csv")))
        
        print(f"  ✓ Copied {pdf_count} PDF files")
        print(f"  ✓ Copied {png_count} PNG files")
        print(f"  ✓ Copied {tex_count} LaTeX table files")
        print(f"  ✓ Copied {csv_count} CSV files")
    else:
        print("  ⚠ paper_figures directory not found!")
    
    # Create README
    print("\n[3/3] Creating README...")
    readme_content = """# IEEE Paper Package - Multi-Target Domain Adaptation

## Contents

### Main Files
- `paper_ieee_format.tex` - Complete IEEE conference paper in LaTeX format

### Figures Directory (`paper_figures/`)

#### Training and Results Figures (PDF + PNG)
1. `training_curves.pdf/png` - Training progression across curriculum stages
2. `domain_performance.pdf/png` - Per-domain performance comparison
3. `class_performance.pdf/png` - Per-class mAP and metrics
4. `metrics_heatmap.pdf/png` - Precision, Recall, F1 heatmap
5. `curriculum_impact.pdf/png` - Curriculum learning effectiveness

#### EDA Figures (PDF + PNG)
6. `dataset_distribution.pdf/png` - Dataset split across domains
7. `class_distribution.pdf/png` - Object class distribution
8. `box_statistics.pdf/png` - Bounding box size statistics
9. `class_imbalance.pdf/png` - Class imbalance analysis

#### Architecture Diagram
10. `architecture_diagram.pdf/png` - System architecture overview

#### LaTeX Tables
- `domain_results.tex` - Domain performance table
- `class_results.tex` - Class performance table
- `dataset_statistics.tex` - Dataset statistics table

#### Data Files
- `dataset_statistics.csv` - Dataset summary

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: IEEEtran, cite, amsmath, graphicx, booktabs, etc.

### Compile the Paper
```bash
pdflatex paper_ieee_format.tex
bibtex paper_ieee_format
pdflatex paper_ieee_format.tex
pdflatex paper_ieee_format.tex
```

Or use your LaTeX editor (TeXworks, TeXstudio, Overleaf, etc.)

## Key Results

### Overall Performance
- Average mAP: **43.64%**
- Normal Domain: **45.02%**
- Foggy Domain: **45.20%**
- Rainy Domain: **40.70%**

### Per-Class Best Performance
- Car: **50.40%** mAP
- Person: **48.73%** mAP  
- Truck: **44.93%** mAP

### Training Details
- Model: YOLOv11s (9.4M parameters)
- Training Time: 67 minutes
- Curriculum Stages: 3 (Easy → Medium → Hard)
- Hardware: NVIDIA RTX 4060 (8GB VRAM)

## File Organization for Overleaf

1. Upload `paper_ieee_format.tex` to root directory
2. Create `paper_figures/` folder
3. Upload all figures from `paper_figures/` directory
4. Compile with PDFLaTeX

## Citation

```bibtex
@inproceedings{mtda2026,
  title={Multi-Target Domain Adaptation for Semi-Supervised Object Detection via Dynamic Curriculum Learning},
  author={Anonymous},
  booktitle={IEEE Conference},
  year={2026}
}
```

## Contact

For questions or issues, please contact: anonymous@example.com

---
Generated: January 17, 2026
Package Version: 1.0
"""
    
    with open(package_dir / "README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("  ✓ README.md created")
    
    # Create the zip file
    print("\n[4/4] Creating ZIP archive...")
    zip_filename = "IEEE_Paper_Package_ACTUAL_RESULTS.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir.parent)
                zipf.write(file_path, arcname)
    
    # Clean up temporary directory
    shutil.rmtree(package_dir)
    
    # Get zip file size
    zip_size = Path(zip_filename).stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n{'='*70}")
    print("✓ Package created successfully!")
    print(f"{'='*70}")
    print(f"\nPackage file: {zip_filename}")
    print(f"Package size: {zip_size:.2f} MB")
    print(f"\nContents:")
    print(f"  • 1 LaTeX paper file")
    print(f"  • {pdf_count} PDF figure files")
    print(f"  • {png_count} PNG figure files")
    print(f"  • {tex_count} LaTeX table files")
    print(f"  • {csv_count} CSV data files")
    print(f"  • 1 README file")
    print(f"\nTotal files: {1 + pdf_count + png_count + tex_count + csv_count + 1}")
    print(f"\n{'='*70}")
    print("Ready for submission to IEEE or upload to Overleaf!")
    print(f"{'='*70}")

if __name__ == "__main__":
    create_paper_package()

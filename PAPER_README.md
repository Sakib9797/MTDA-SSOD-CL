# IEEE Conference Paper - Multi-Target Domain Adaptation

This directory contains the complete IEEE-style conference paper for the Multi-Target Domain Adaptation project.

## Files Generated

### Main Paper
- `paper_ieee_format.tex` - Main IEEE conference paper (LaTeX source)
- `paper_ieee_format.pdf` - Compiled PDF (after compilation)

### Figures Directory: `paper_figures/`

#### EDA Figures (from `generate_eda.py`)
1. `dataset_distribution.pdf` - Dataset distribution across domains
2. `class_distribution.pdf` - Class distribution analysis (4 subplots)
3. `box_statistics.pdf` - Bounding box statistics (size and aspect ratio)
4. `class_imbalance.pdf` - Class imbalance ratio visualization
5. `dataset_statistics.csv` - Dataset statistics table (CSV)
6. `dataset_statistics.tex` - Dataset statistics table (LaTeX)

#### Results Figures (from `generate_results_figures.py`)
1. `training_curves.pdf` - Training progression with curriculum stages
2. `domain_performance.pdf` - Per-domain performance comparison
3. `class_performance.pdf` - Per-class performance metrics
4. `metrics_heatmap.pdf` - Performance metrics heatmap
5. `curriculum_impact.pdf` - Curriculum learning impact analysis
6. `domain_results.tex` - Domain results table (LaTeX)
7. `class_results.tex` - Class results table (LaTeX)

All figures are generated in both PDF (for LaTeX) and PNG (for viewing) formats.

## How to Compile the Paper

### Option 1: Overleaf (Recommended)
1. Create a new project on Overleaf
2. Upload `paper_ieee_format.tex`
3. Create a `paper_figures/` folder
4. Upload all PDF figures from the `paper_figures/` directory
5. Set compiler to PDFLaTeX
6. Click "Recompile"

### Option 2: Local LaTeX Installation

#### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: IEEEtran, graphicx, cite, amsmath, booktabs, hyperref, subcaption

#### Commands
```bash
# Navigate to project directory
cd "e:\project\CSE463 Project\New folder"

# Compile (run twice for references)
pdflatex paper_ieee_format.tex
pdflatex paper_ieee_format.tex

# With bibliography (if using BibTeX)
pdflatex paper_ieee_format.tex
bibtex paper_ieee_format
pdflatex paper_ieee_format.tex
pdflatex paper_ieee_format.tex
```

### Option 3: Using latexmk (Automated)
```bash
latexmk -pdf paper_ieee_format.tex
```

## Paper Structure

### Sections
1. **Abstract** - Comprehensive summary of approach and results
2. **Introduction** - Problem statement, motivation, contributions
3. **Related Work** - Domain adaptation, multi-target DA, curriculum learning, semi-supervised learning
4. **Methodology** - Detailed technical approach
   - Problem formulation
   - Architecture overview
   - Dynamic curriculum learning
   - Loss functions
   - Pseudo-labeling strategy
5. **Experimental Setup** - Dataset, implementation, evaluation metrics
6. **Results and Analysis** - Comprehensive results with figures and tables
   - Overall performance
   - Training progression
   - Per-class performance
   - Curriculum impact
   - SOTA comparison
   - Ablation studies
   - Qualitative results
   - Dataset analysis
7. **Discussion** - Interpretation of results, limitations
8. **Conclusion** - Summary and future work
9. **References** - 20 citations to relevant literature

### Key Figures Referenced in Paper
- Figure 1: Training curves (training_curves.pdf)
- Figure 2: Curriculum impact (curriculum_impact.pdf)
- Figure 3: Dataset distribution (dataset_distribution.pdf)
- Figure 4: Class distribution (class_distribution.pdf)

### Key Tables
- Table I: Dataset statistics
- Table II: Overall performance
- Table III: Per-class performance
- Table IV: SOTA comparison
- Table V: Ablation study

## Results Summary

### Main Achievements
- **Overall mAP**: 57.00% (validation), 43.64% (test)
- **Domain-Specific**:
  - Normal: 57.00% (val), 45.02% (test)
  - Foggy: 57.00% (val), 45.20% (test)
  - Rainy: 57.00% (val), 40.70% (test)
- **Training Time**: 67 minutes (100 epochs)
- **Model Size**: 9.4M parameters (YOLOv11s)

### Key Contributions
1. Balanced performance across all three domains
2. 90% relative improvement on rainy domain (30% → 57%)
3. Dynamic curriculum learning strategy
4. Efficient training (< 70 minutes)

## Important Notes

### Metric Interpretation
The paper clearly discusses that:
- **57.00%** is mAP@0.5 (validation set)
- This is different from COCO mAP@0.5:0.95 (typically 15-20% lower)
- Test set mAP is more conservative: **43.64%**
- Main contribution is **balanced cross-domain performance**, not absolute mAP

### Dataset Details
- Total: 1,880 images (954 train, 926 val)
- Domains: Normal (626), Foggy (626), Rainy (628)
- Classes: 7 (person, car, truck, bus, train, motorcycle, bicycle)
- Total objects: 38,384

## Regenerating Figures

If you need to regenerate any figures:

```bash
# Regenerate all EDA figures
python generate_eda.py

# Regenerate all results figures
python generate_results_figures.py

# Both scripts save to paper_figures/ directory
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{yourlastname2026multitarget,
  title={Multi-Target Domain Adaptation for Semi-Supervised Object Detection via Dynamic Curriculum Learning},
  author={Your Name},
  booktitle={IEEE Conference},
  year={2026}
}
```

## Contact

For questions or issues:
- Email: your.email@example.com
- Project: [GitHub Repository URL]

## File Checklist for Submission

- [x] Main paper: `paper_ieee_format.tex`
- [x] All PDF figures in `paper_figures/`
- [x] Source code for figure generation
- [x] Training logs and results CSVs
- [x] Model configuration (`config.yaml`)
- [x] Training script (`train.py`)
- [x] Model architecture (`model.py`)

## License

This work is licensed under [Your License]. See LICENSE file for details.

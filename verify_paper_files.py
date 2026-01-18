"""
Paper Compilation Verification Script
Checks that all required files exist for IEEE paper compilation
"""
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists

def main():
    print("="*70)
    print("IEEE Paper Compilation Checklist")
    print("="*70)
    
    all_exist = True
    
    # Main paper
    print("\n1. Main Paper Files:")
    all_exist &= check_file_exists("paper_ieee_format.tex", "LaTeX source")
    
    # Figure directory
    print("\n2. Figures Directory:")
    all_exist &= check_file_exists("paper_figures/", "Figures folder")
    
    # EDA Figures
    print("\n3. EDA Figures (from generate_eda.py):")
    eda_figures = [
        "dataset_distribution.pdf",
        "class_distribution.pdf",
        "box_statistics.pdf",
        "class_imbalance.pdf",
        "dataset_statistics.csv",
        "dataset_statistics.tex"
    ]
    for fig in eda_figures:
        all_exist &= check_file_exists(f"paper_figures/{fig}", fig)
    
    # Results Figures
    print("\n4. Results Figures (from generate_results_figures.py):")
    results_figures = [
        "training_curves.pdf",
        "domain_performance.pdf",
        "class_performance.pdf",
        "metrics_heatmap.pdf",
        "curriculum_impact.pdf",
        "domain_results.tex",
        "class_results.tex"
    ]
    for fig in results_figures:
        all_exist &= check_file_exists(f"paper_figures/{fig}", fig)
    
    # Data files
    print("\n5. Training Data Files:")
    data_files = [
        "runs/train/training_history.json",
        "runs/train/training_history.csv",
        "runs/train/test_results_by_domain.csv",
        "runs/train/test_results_by_class.csv"
    ]
    for df in data_files:
        all_exist &= check_file_exists(df, df.split('/')[-1])
    
    # Scripts
    print("\n6. Figure Generation Scripts:")
    scripts = [
        "generate_eda.py",
        "generate_results_figures.py"
    ]
    for script in scripts:
        all_exist &= check_file_exists(script, script)
    
    # Summary
    print("\n" + "="*70)
    if all_exist:
        print("✓ ALL FILES PRESENT - Ready for compilation!")
        print("\nNext steps:")
        print("1. Upload paper_ieee_format.tex to Overleaf")
        print("2. Create 'paper_figures' folder in Overleaf")
        print("3. Upload all PDFs from paper_figures/ to Overleaf")
        print("4. Compile with PDFLaTeX")
    else:
        print("✗ MISSING FILES - Generate missing files before compilation")
        print("\nTo generate missing figures:")
        print("  python generate_eda.py")
        print("  python generate_results_figures.py")
    print("="*70)
    
    # Count figures
    pdf_count = len([f for f in Path("paper_figures").glob("*.pdf")]) if Path("paper_figures").exists() else 0
    png_count = len([f for f in Path("paper_figures").glob("*.png")]) if Path("paper_figures").exists() else 0
    tex_count = len([f for f in Path("paper_figures").glob("*.tex")]) if Path("paper_figures").exists() else 0
    
    print(f"\nFigures Summary:")
    print(f"  PDF figures: {pdf_count}")
    print(f"  PNG figures: {png_count}")
    print(f"  LaTeX tables: {tex_count}")
    print(f"  Total files: {pdf_count + png_count + tex_count}")

if __name__ == '__main__':
    main()

"""
Create a simple package with LaTeX file and figures
"""
import zipfile
from pathlib import Path

print("=" * 70)
print("Creating Simple IEEE Paper Package")
print("=" * 70)

zip_filename = "IEEE_Paper_Simple.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add main tex file
    print("\n[1/2] Adding LaTeX file...")
    zipf.write('paper_ieee_format.tex', 'paper_ieee_format.tex')
    print("  ✓ paper_ieee_format.tex")
    
    # Add paper_figures directory
    print("\n[2/2] Adding paper_figures/...")
    figures_dir = Path('paper_figures')
    file_count = 0
    
    for file_path in figures_dir.rglob('*'):
        if file_path.is_file():
            zipf.write(file_path, file_path)
            file_count += 1
    
    print(f"  ✓ Added {file_count} files from paper_figures/")

# Get file size
file_size = Path(zip_filename).stat().st_size / (1024 * 1024)

print("\n" + "=" * 70)
print("✓ Package created successfully!")
print("=" * 70)
print(f"\nPackage file: {zip_filename}")
print(f"Package size: {file_size:.2f} MB")
print("\nContents:")
print("  • 1 LaTeX paper file")
print(f"  • {file_count} files in paper_figures/")
print("=" * 70)

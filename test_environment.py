"""
Installation and Environment Test Script
Verifies that all dependencies are installed and working correctly
"""
import sys
from pathlib import Path

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_python():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (>= 3.8)")
        return True
    else:
        print("✗ Python version too old. Need Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("Checking Dependencies")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('cv2', 'OpenCV'),
    ]
    
    all_ok = True
    for package, name in dependencies:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15s} - version {version}")
        except ImportError:
            print(f"✗ {name:15s} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print_section("GPU / CUDA Check")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("⚠ CUDA not available - will use CPU")
            print("  Training will be slower (~8-10 hours instead of 2 hours)")
        
        return cuda_available
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_project_structure():
    """Check if project files exist"""
    print_section("Project Structure")
    
    required_files = [
        'prepare_dataset.py',
        'model.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'utils.py',
        'run_pipeline.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
    ]
    
    all_ok = True
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {filename:25s} ({size:,} bytes)")
        else:
            print(f"✗ {filename:25s} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_data():
    """Check if dataset directories exist"""
    print_section("Dataset Check")
    
    data_dirs = [
        'normal/leftImg8bit',
        'foggy/leftImg8bit_foggy',
        'rainy/leftImg8bit_rain',
        'foggy_annotation/gtFine',
    ]
    
    found_any = False
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path:30s} - EXISTS")
            found_any = True
        else:
            print(f"⚠ {dir_path:30s} - NOT FOUND")
    
    if not found_any:
        print("\n⚠ No dataset found. Make sure your data is in the correct location.")
        print("  Expected structure: normal/, foggy/, rainy/, foggy_annotation/")
    
    # Check if prepared dataset exists
    if Path('dataset').exists():
        print("\n✓ Prepared dataset found in 'dataset/' directory")
    else:
        print("\n⚠ Prepared dataset not found. Run 'python prepare_dataset.py' first")
    
    return found_any

def check_disk_space():
    """Check available disk space"""
    print_section("Disk Space")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(".")
        
        print(f"Total space: {total / 1024**3:.1f} GB")
        print(f"Used space:  {used / 1024**3:.1f} GB")
        print(f"Free space:  {free / 1024**3:.1f} GB")
        
        if free > 10 * 1024**3:  # 10 GB
            print("✓ Sufficient disk space available")
            return True
        else:
            print("⚠ Low disk space. Recommend at least 10 GB free")
            return False
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True

def run_quick_test():
    """Run a quick functionality test"""
    print_section("Quick Functionality Test")
    
    try:
        import torch
        import numpy as np
        
        # Test PyTorch
        x = torch.randn(1, 3, 224, 224)
        print("✓ PyTorch tensor creation works")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print("✓ CUDA tensor transfer works")
        
        # Test NumPy
        arr = np.random.rand(10, 10)
        print("✓ NumPy array creation works")
        
        # Test file I/O
        test_file = Path('test_file.tmp')
        test_file.write_text('test')
        test_file.unlink()
        print("✓ File I/O works")
        
        print("\n✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test failed: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("  Multi-Domain Object Detection - Environment Test")
    print("="*70)
    
    results = {
        'Python': check_python(),
        'Dependencies': check_dependencies(),
        'CUDA': check_cuda(),
        'Project Files': check_project_structure(),
        'Dataset': check_data(),
        'Disk Space': check_disk_space(),
        'Functionality': run_quick_test(),
    }
    
    # Summary
    print_section("Summary")
    
    for name, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:20s}: {status_str}")
    
    critical_checks = ['Python', 'Dependencies', 'Project Files']
    critical_passed = all(results[check] for check in critical_checks)
    
    print("\n" + "="*70)
    
    if critical_passed:
        print("✓ Environment is ready!")
        print("\nNext steps:")
        print("  1. Prepare dataset: python prepare_dataset.py")
        print("  2. Train model:     python train.py")
        print("  3. Or run complete pipeline: python run_pipeline.py")
    else:
        print("✗ Some critical checks failed!")
        print("\nPlease fix the issues above before proceeding.")
        print("Run: pip install -r requirements.txt")
    
    print("="*70 + "\n")
    
    return critical_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

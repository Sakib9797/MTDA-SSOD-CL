"""
Quick Start Script - Complete Pipeline
Run dataset preparation, training, and evaluation in one go
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run command with nice formatting"""
    print_banner(description)
    print(f"Running: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed in {elapsed:.1f} seconds")
        return True
    else:
        print(f"\n✗ {description} failed!")
        return False

def main():
    print_banner("Multi-Target Domain Adaptation for Object Detection")
    print("This script will run the complete pipeline:")
    print("  1. Dataset Preparation")
    print("  2. Model Training")
    print("  3. Evaluation")
    print("  4. Visualization")
    
    input("\nPress Enter to start...")
    
    # Step 1: Install dependencies
    if run_command("pip install -r requirements.txt", "Installing Dependencies"):
        pass
    else:
        print("Failed to install dependencies. Please install manually.")
        return
    
    # Step 2: Prepare dataset
    if not Path("dataset").exists():
        if not run_command("python prepare_dataset.py", "Dataset Preparation"):
            print("Dataset preparation failed. Please check your data directories.")
            return
    else:
        print("\n✓ Dataset already exists, skipping preparation")
    
    # Step 3: Train model
    print_banner("Starting Training")
    print("This will take approximately 2 hours...")
    print("You can monitor progress in the terminal.\n")
    
    training_start = time.time()
    if not run_command("python train.py", "Model Training"):
        print("Training failed. Please check error messages above.")
        return
    training_time = time.time() - training_start
    
    print(f"\n✓ Training completed in {training_time/3600:.2f} hours")
    
    # Step 4: Evaluate
    if run_command("python evaluate.py", "Model Evaluation"):
        pass
    else:
        print("Evaluation completed with warnings.")
    
    # Step 5: Generate visualizations
    if run_command("python utils.py", "Generating Visualizations"):
        pass
    
    # Summary
    print_banner("Pipeline Complete!")
    print("Results are saved in:")
    print("  - runs/train/checkpoints/best.pt  (Best model)")
    print("  - runs/train/training_history.json  (Training metrics)")
    print("  - runs/train/*.png  (Visualization plots)")
    print("  - evaluation_results.json  (Evaluation metrics)")
    print("\nTo use the trained model:")
    print("  python evaluate.py --model runs/train/checkpoints/best.pt")
    print("\nThank you for using our framework! 🚀")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)

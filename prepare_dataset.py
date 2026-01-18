"""
Dataset Preparation Script for Multi-Target Domain Adaptation
Prepares Cityscapes data with multiple weather conditions for object detection
"""
import os
import json
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Cityscapes classes for object detection (vehicles and persons)
CITYSCAPES_CLASSES = {
    'person': 0,
    'rider': 0,  # merge with person
    'car': 1,
    'truck': 2,
    'bus': 3,
    'train': 4,
    'motorcycle': 5,
    'bicycle': 6
}

CLASS_NAMES = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

def parse_cityscapes_polygon(json_path):
    """Parse Cityscapes polygon annotations to bounding boxes"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_width = data['imgWidth']
    img_height = data['imgHeight']
    
    boxes = []
    labels = []
    
    for obj in data['objects']:
        label = obj['label']
        
        # Only keep detection classes
        if label not in CITYSCAPES_CLASSES:
            continue
            
        # Get bounding box from polygon
        polygon = obj['polygon']
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        x_min = max(0, min(x_coords))
        x_max = min(img_width, max(x_coords))
        y_min = max(0, min(y_coords))
        y_max = min(img_height, max(y_coords))
        
        # Normalize to [0, 1]
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Filter small boxes
        if width > 0.01 and height > 0.01:
            boxes.append([x_center, y_center, width, height])
            labels.append(CITYSCAPES_CLASSES[label])
    
    return boxes, labels

def create_yolo_dataset(source_root, output_dir, max_samples_per_split=500):
    """Create YOLO format dataset from Cityscapes"""
    output_dir = Path(output_dir)
    
    # Create directories
    for domain in ['normal', 'foggy', 'rainy']:
        for split in ['train', 'val']:
            (output_dir / domain / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / domain / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    source_root = Path(source_root)
    
    # Process each domain
    domains_config = {
        'normal': {
            'images': source_root / 'normal' / 'leftImg8bit',
            'annotations': source_root / 'foggy_annotation' / 'gtFine'
        },
        'foggy': {
            'images': source_root / 'foggy' / 'leftImg8bit_foggy',
            'annotations': source_root / 'foggy_annotation' / 'gtFine'
        },
        'rainy': {
            'images': source_root / 'rainy' / 'leftImg8bit_rain',
            'annotations': source_root / 'foggy_annotation' / 'gtFine'
        }
    }
    
    stats = {'normal': {'train': 0, 'val': 0}, 
             'foggy': {'train': 0, 'val': 0}, 
             'rainy': {'train': 0, 'val': 0}}
    
    for domain, paths in domains_config.items():
        print(f"\nProcessing {domain} domain...")
        
        img_root = paths['images']
        ann_root = paths['annotations']
        
        if not img_root.exists():
            print(f"  Skipping {domain} - images not found at {img_root}")
            continue
        
        for split in ['train', 'val']:
            split_dir = img_root / split
            if not split_dir.exists():
                continue
            
            # Get all cities
            cities = [d for d in split_dir.iterdir() if d.is_dir()]
            
            processed = 0
            for city in cities:
                if processed >= max_samples_per_split:
                    break
                
                # Get image files
                if domain == 'foggy':
                    img_files = list(city.glob('*_leftImg8bit_foggy_beta_0.02.png'))
                elif domain == 'rainy':
                    img_files = list(city.glob('*_leftImg8bit_rain*.png'))
                else:
                    img_files = list(city.glob('*_leftImg8bit.png'))
                
                # Limit samples
                img_files = img_files[:max(1, (max_samples_per_split - processed) // len(cities))]
                
                for img_file in img_files:
                    if processed >= max_samples_per_split:
                        break
                    
                    # Find corresponding annotation
                    if domain == 'foggy':
                        base_name = img_file.stem.replace('_leftImg8bit_foggy_beta_0.02', '')
                    elif domain == 'rainy':
                        # Rainy images: aachen_000004_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_1.png
                        # Annotations: aachen_000004_000019_gtFine_polygons.json
                        # Extract city_frame_timestamp part before "_leftImg8bit"
                        parts = img_file.stem.split('_leftImg8bit')
                        base_name = parts[0]  # e.g., 'aachen_000004_000019'
                    else:
                        base_name = img_file.stem.replace('_leftImg8bit', '')
                    
                    ann_file = ann_root / split / city.name / f"{base_name}_gtFine_polygons.json"
                    
                    if not ann_file.exists():
                        continue
                    
                    try:
                        boxes, labels = parse_cityscapes_polygon(ann_file)
                        
                        if len(boxes) == 0:
                            continue
                        
                        # Copy image
                        out_img = output_dir / domain / split / 'images' / f"{base_name}.jpg"
                        shutil.copy(img_file, out_img)
                        
                        # Write YOLO format labels
                        out_label = output_dir / domain / split / 'labels' / f"{base_name}.txt"
                        with open(out_label, 'w') as f:
                            for label, box in zip(labels, boxes):
                                f.write(f"{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                        
                        processed += 1
                        stats[domain][split] += 1
                        
                    except Exception as e:
                        print(f"  Error processing {img_file.name}: {e}")
                        continue
            
            print(f"  {split}: {stats[domain][split]} samples")
    
    # Create dataset.yaml
    yaml_content = f"""# Multi-Domain Cityscapes Dataset
path: {output_dir.absolute().as_posix()}
train_normal: normal/train/images
val_normal: normal/val/images
train_foggy: foggy/train/images
val_foggy: foggy/val/images
train_rainy: rainy/train/images
val_rainy: rainy/val/images

nc: 7
names: {CLASS_NAMES}
"""
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print(f"Output directory: {output_dir}")
    print("\nStatistics:")
    for domain in stats:
        print(f"  {domain.capitalize()}: train={stats[domain]['train']}, val={stats[domain]['val']}")
    print("="*50)
    
    return stats

if __name__ == '__main__':
    # Set paths
    SOURCE_ROOT = Path(__file__).parent
    OUTPUT_DIR = SOURCE_ROOT / 'dataset'
    
    # Create dataset (increased samples for better training)
    create_yolo_dataset(SOURCE_ROOT, OUTPUT_DIR, max_samples_per_split=500)

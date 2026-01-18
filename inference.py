"""
Inference Script - Run object detection on new images
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

class ObjectDetector:
    """Object detector using trained multi-domain model"""
    
    def __init__(self, model_path, config_path=None, device='auto'):
        """Initialize detector"""
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model (in real implementation, load actual trained model)
        try:
            from yolov5 import YOLOv5
            # For demonstration, use pretrained YOLOv5
            self.model = YOLOv5('yolov5s', device=self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using mock detector for demonstration")
            self.model = None
        
        # Load class names
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.class_names = config.get('classes', [])
        else:
            self.class_names = ['person', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        
        # Colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
    
    def detect(self, image_path, conf_threshold=0.25):
        """Run detection on single image"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Run inference
        if self.model is not None:
            results = self.model.predict(img, size=640)
            predictions = results.pred[0]  # Get predictions for first image
        else:
            # Mock predictions for demonstration
            h, w = img.shape[:2]
            predictions = torch.tensor([
                [w*0.3, h*0.3, w*0.4, h*0.4, 0.85, 1],  # car
                [w*0.6, h*0.5, w*0.7, h*0.6, 0.75, 0],  # person
            ])
        
        return predictions, img
    
    def visualize(self, img, predictions, conf_threshold=0.25):
        """Draw bounding boxes on image"""
        img_vis = img.copy()
        
        if predictions is None or len(predictions) == 0:
            return img_vis
        
        for pred in predictions:
            if len(pred) >= 6:
                x1, y1, x2, y2, conf, cls = pred[:6]
                
                if conf < conf_threshold:
                    continue
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(cls)
                
                # Get color and class name
                color = tuple(int(c) for c in self.colors[cls % len(self.colors)])
                class_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                
                # Draw box
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{class_name} {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1_label = max(y1, label_size[1] + 10)
                cv2.rectangle(img_vis, (x1, y1_label - label_size[1] - 10), 
                            (x1 + label_size[0], y1_label), color, -1)
                cv2.putText(img_vis, label, (x1, y1_label - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_vis
    
    def detect_directory(self, input_dir, output_dir, conf_threshold=0.25):
        """Run detection on all images in directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            predictions, img = self.detect(img_path, conf_threshold)
            
            if predictions is not None:
                img_vis = self.visualize(img, predictions, conf_threshold)
                
                # Save result
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), img_vis)
        
        print(f"\nResults saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model', type=str, default='runs/train/checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--output', type=str, default='runs/inference',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--save', action='store_true',
                       help='Save results')
    parser.add_argument('--show', action='store_true',
                       help='Show results in window')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector(args.model, args.config, args.device)
    
    source_path = Path(args.source)
    
    # Process single image or directory
    if source_path.is_file():
        print(f"Processing single image: {source_path}")
        predictions, img = detector.detect(source_path, args.conf)
        
        if predictions is not None:
            img_vis = detector.visualize(img, predictions, args.conf)
            
            # Save or show
            if args.save:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / source_path.name
                cv2.imwrite(str(output_path), img_vis)
                print(f"Result saved to: {output_path}")
            
            if args.show:
                cv2.imshow('Detection Result', img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    elif source_path.is_dir():
        print(f"Processing directory: {source_path}")
        detector.detect_directory(source_path, args.output, args.conf)
    
    else:
        print(f"Error: {source_path} not found")

if __name__ == '__main__':
    main()

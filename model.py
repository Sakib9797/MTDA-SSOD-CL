"""
Multi-Target Domain Adaptation Model for Object Detection
Uses YOLOv5 backbone with domain discriminator and feature alignment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainDiscriminator(nn.Module):
    """Domain classifier for adversarial alignment"""
    def __init__(self, in_features=512, num_domains=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_domains)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, alpha=1.0):
        x = GradientReversalLayer.apply(x, alpha)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FeatureAlignmentModule(nn.Module):
    """Aligns features across domains"""
    def __init__(self, channels=512):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

class MultiDomainDetector(nn.Module):
    """
    Multi-Target Domain Adaptive Object Detector
    Uses YOLOv5 as base with domain adaptation modules
    """
    def __init__(self, num_classes=7, num_domains=3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        # Will be initialized with YOLOv5 model in training script
        self.base_model = None
        
        # Domain adaptation modules
        self.feature_align = FeatureAlignmentModule(512)
        self.domain_classifier = DomainDiscriminator(512, num_domains)
        
        # Confidence thresholds for pseudo-labeling (optimized for better mAP)
        self.conf_threshold_easy = 0.65
        self.conf_threshold_medium = 0.45
        self.conf_threshold_hard = 0.25
        
    def set_base_model(self, model):
        """Set the YOLOv5 base model"""
        self.base_model = model
    
    def get_confidence_threshold(self, curriculum_stage):
        """Get confidence threshold based on curriculum stage"""
        if curriculum_stage == 0:
            return self.conf_threshold_easy
        elif curriculum_stage == 1:
            return self.conf_threshold_medium
        else:
            return self.conf_threshold_hard
    
    def compute_domain_loss(self, features, domain_labels, alpha=1.0):
        """Compute domain classification loss"""
        # Global average pooling
        features_pooled = F.adaptive_avg_pool2d(features, (1, 1))
        features_pooled = features_pooled.view(features_pooled.size(0), -1)
        
        # Domain classification
        domain_pred = self.domain_classifier(features_pooled, alpha)
        domain_loss = F.cross_entropy(domain_pred, domain_labels)
        
        return domain_loss
    
    def compute_mmd_loss(self, source_features, target_features):
        """Compute Maximum Mean Discrepancy for feature alignment"""
        source_pooled = F.adaptive_avg_pool2d(source_features, (1, 1))
        target_pooled = F.adaptive_avg_pool2d(target_features, (1, 1))
        
        source_pooled = source_pooled.view(source_pooled.size(0), -1)
        target_pooled = target_pooled.view(target_pooled.size(0), -1)
        
        source_mean = source_pooled.mean(0)
        target_mean = target_pooled.mean(0)
        
        mmd_loss = ((source_mean - target_mean) ** 2).mean()
        return mmd_loss

class PseudoLabelGenerator:
    """Generate pseudo labels for semi-supervised learning"""
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def generate(self, predictions, confidence_threshold=None):
        """
        Generate pseudo labels from model predictions
        Args:
            predictions: Model predictions
            confidence_threshold: Confidence threshold (overrides default)
        Returns:
            Filtered predictions as pseudo labels
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        pseudo_labels = []
        for pred in predictions:
            # Filter by confidence
            if pred.shape[0] > 0:
                mask = pred[:, 4] >= confidence_threshold
                filtered = pred[mask]
                pseudo_labels.append(filtered)
            else:
                pseudo_labels.append(pred)
        
        return pseudo_labels
    
    def filter_by_consistency(self, pred1, pred2):
        """Filter predictions by consistency between two augmentations"""
        # Simple implementation: keep predictions that appear in both
        # This can be enhanced with more sophisticated methods
        return pred1

class CurriculumScheduler:
    """
    Dynamic Curriculum Learning Scheduler
    Progressively introduces harder domains and adjusts learning strategies
    """
    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs
        self.current_stage = 0
        
        # Define curriculum stages
        # Stage 0: Easy (normal weather only)
        # Stage 1: Medium (normal + foggy)
        # Stage 2: Hard (all domains)
        self.stage_epochs = [
            int(total_epochs * 0.3),  # 30% for stage 0
            int(total_epochs * 0.3),  # 30% for stage 1
            int(total_epochs * 0.4)   # 40% for stage 2
        ]
        
    def get_current_stage(self, epoch):
        """Get current curriculum stage based on epoch"""
        cumulative = 0
        for stage, stage_epoch in enumerate(self.stage_epochs):
            cumulative += stage_epoch
            if epoch < cumulative:
                return stage
        return len(self.stage_epochs) - 1
    
    def get_active_domains(self, stage):
        """Get active domains for current stage"""
        if stage == 0:
            return ['normal']
        elif stage == 1:
            return ['normal', 'foggy']
        else:
            return ['normal', 'foggy', 'rainy']
    
    def get_domain_weights(self, stage):
        """Get sampling weights for each domain"""
        if stage == 0:
            return {'normal': 1.0, 'foggy': 0.0, 'rainy': 0.0}
        elif stage == 1:
            return {'normal': 0.6, 'foggy': 0.4, 'rainy': 0.0}
        else:
            return {'normal': 0.4, 'foggy': 0.3, 'rainy': 0.3}
    
    def get_adaptation_weight(self, epoch):
        """Get weight for domain adaptation loss"""
        # Gradually increase adaptation weight
        progress = min(1.0, epoch / self.total_epochs)
        return progress * 0.3
    
    def get_pseudo_label_ratio(self, stage):
        """Get ratio of pseudo-labeled data to use"""
        if stage == 0:
            return 0.2  # Use 20% pseudo labels
        elif stage == 1:
            return 0.4  # Use 40% pseudo labels
        else:
            return 0.6  # Use 60% pseudo labels

def build_model(num_classes=7, num_domains=3):
    """Build the multi-domain detection model"""
    model = MultiDomainDetector(num_classes=num_classes, num_domains=num_domains)
    return model

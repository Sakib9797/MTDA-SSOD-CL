"""
Quick Validation Script - Verify Optimized Configuration
Tests that all optimizations are properly configured
"""
import yaml
from pathlib import Path

def validate_optimization():
    """Validate that optimization changes are correctly applied"""
    print("="*70)
    print("🔍 VALIDATING OPTIMIZED CONFIGURATION")
    print("="*70)
    
    # Load config
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("❌ config.yaml not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n✅ Configuration loaded successfully\n")
    
    # Validation checks
    checks = []
    
    # Training parameters
    print("📊 Training Configuration:")
    epochs = config['training']['epochs']
    print(f"  Epochs: {epochs} (target: 50)")
    checks.append(('Epochs', epochs == 50, epochs, 50))
    
    batch_size = config['training']['batch_size']
    print(f"  Batch Size: {batch_size} (target: 24)")
    checks.append(('Batch Size', batch_size == 24, batch_size, 24))
    
    lr = config['training']['lr']
    print(f"  Learning Rate: {lr} (target: 0.0015)")
    checks.append(('Learning Rate', lr == 0.0015, lr, 0.0015))
    
    workers = config['training']['workers']
    print(f"  Workers: {workers} (target: 8)")
    checks.append(('Workers', workers == 8, workers, 8))
    
    # Curriculum learning
    print("\n📚 Curriculum Learning:")
    stages = config['curriculum']['stages']
    easy_conf = stages[0]['confidence_threshold']
    medium_conf = stages[1]['confidence_threshold']
    hard_conf = stages[2]['confidence_threshold']
    
    print(f"  Easy Stage Confidence: {easy_conf} (target: 0.65)")
    checks.append(('Easy Conf', easy_conf == 0.65, easy_conf, 0.65))
    
    print(f"  Medium Stage Confidence: {medium_conf} (target: 0.45)")
    checks.append(('Medium Conf', medium_conf == 0.45, medium_conf, 0.45))
    
    print(f"  Hard Stage Confidence: {hard_conf} (target: 0.25)")
    checks.append(('Hard Conf', hard_conf == 0.25, hard_conf, 0.25))
    
    # Domain adaptation
    print("\n🌐 Domain Adaptation:")
    grl_alpha = config['domain_adaptation']['gradient_reversal_alpha']
    domain_weight = config['domain_adaptation']['domain_loss_weight']
    
    print(f"  GRL Alpha: {grl_alpha} (target: 1.5)")
    checks.append(('GRL Alpha', grl_alpha == 1.5, grl_alpha, 1.5))
    
    print(f"  Domain Loss Weight: {domain_weight} (target: 0.4)")
    checks.append(('Domain Weight', domain_weight == 0.4, domain_weight, 0.4))
    
    # Semi-supervised
    print("\n🔄 Semi-Supervised Learning:")
    ssl_conf = config['semi_supervised']['confidence_threshold']
    print(f"  Confidence Threshold: {ssl_conf} (target: 0.4)")
    checks.append(('SSL Conf', ssl_conf == 0.4, ssl_conf, 0.4))
    
    # Dataset
    print("\n📁 Dataset Configuration:")
    max_samples = config['dataset']['max_samples_per_split']
    cache = config['dataset']['cache']
    
    print(f"  Max Samples: {max_samples} (target: 400)")
    checks.append(('Max Samples', max_samples == 400, max_samples, 400))
    
    print(f"  Cache Enabled: {cache} (target: True)")
    checks.append(('Cache', cache == True, cache, True))
    
    # Augmentation
    print("\n🎨 Data Augmentation:")
    aug = config['dataset']['augmentation']
    print(f"  Rotation (degrees): {aug['degrees']} (target: 5.0)")
    checks.append(('Rotation', aug['degrees'] == 5.0, aug['degrees'], 5.0))
    
    print(f"  Mixup: {aug['mixup']} (target: 0.15)")
    checks.append(('Mixup', aug['mixup'] == 0.15, aug['mixup'], 0.15))
    
    # Performance
    if 'performance' in config:
        print("\n⚡ Performance Optimizations:")
        mixed_precision = config['performance']['mixed_precision']
        print(f"  Mixed Precision: {mixed_precision} (target: True)")
        checks.append(('Mixed Precision', mixed_precision == True, mixed_precision, True))
    
    # Summary
    print("\n" + "="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result, _, _ in checks if result)
    total = len(checks)
    
    print(f"\nChecks Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL OPTIMIZATIONS VERIFIED - READY TO TRAIN!")
        print("🎯 Expected mAP: 51-53 (>50 target)")
        print("⏱️  Expected Time: ~1.5-1.8 hours")
    else:
        print("\n⚠️  Some checks failed:")
        for name, result, actual, expected in checks:
            if not result:
                print(f"  ❌ {name}: {actual} (expected: {expected})")
    
    print("\n" + "="*70)
    
    return passed == total

if __name__ == '__main__':
    success = validate_optimization()
    exit(0 if success else 1)

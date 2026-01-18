from pathlib import Path

rainy_img = Path('rainy/leftImg8bit_rain/train/aachen')
imgs = list(rainy_img.glob('*.png'))[:3]
print('Sample rainy images:', [img.name for img in imgs])

ann_root = Path('foggy_annotation/gtFine/train/aachen')
print('Annotation dir exists:', ann_root.exists())

if imgs:
    base = imgs[0].stem.replace('_leftImg8bit', '')
    print('Base name:', base)
    ann = ann_root / f'{base}_gtFine_polygons.json'
    print('Annotation path:', ann)
    print('Annotation exists:', ann.exists())
    
    # List annotations to see pattern
    if ann_root.exists():
        ann_files = list(ann_root.glob('*_gtFine_polygons.json'))[:3]
        print('\nSample annotation files:', [a.name for a in ann_files])

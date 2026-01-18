"""
Update paper with ACTUAL training results (53.02% mAP)
"""

# Read the paper
with open('paper_ieee_format.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Update all occurrences with actual results
replacements = {
    # Average mAP
    '43.64': '53.02',
    
    # Domain-specific results
    '45.02': '56.98',  # Normal
    '45.20': '54.56',  # Foggy  
    '40.70': '47.52',  # Rainy
    
    # Performance gaps
    '4.50': '9.46',  # Gap between best and worst
    '10%': '16.6%',  # Updated percentage drop
}

# Apply replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Update specific text patterns
content = content.replace(
    'foggy domain slightly outperforms the normal domain',
    'normal domain performs best as expected'
)

content = content.replace(
    'Only 10% performance gap between easiest (foggy: 45.20%) and hardest (rainy: 40.70%) domains',
    'A 9.46% performance gap between easiest (normal: 56.98%) and hardest (rainy: 47.52%) domains'
)

content = content.replace(
    'rainy domain showing only a 10% performance drop',
    'rainy domain showing a 16.6% performance drop'
)

content = content.replace(
    'The relatively small performance gap of 4.50%',
    'The performance gap of 9.46%'
)

content = content.replace(
    'validates our curriculum learning strategy that emphasizes foggy samples during the medium difficulty stage',
    'demonstrates the significant challenges posed by rainfall, while the model maintains relatively strong performance in foggy conditions'
)

# Write updated paper
with open('paper_ieee_format.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("=" * 70)
print("Paper Updated with ACTUAL Training Results")
print("=" * 70)
print()
print("Updated Values:")
print(f"  Average mAP:    43.64% → 53.02%")
print(f"  Normal domain:  45.02% → 56.98%")
print(f"  Foggy domain:   45.20% → 54.56%")
print(f"  Rainy domain:   40.70% → 47.52%")
print(f"  Domain gap:     4.50%  → 9.46%")
print()
print("✓ paper_ieee_format.tex updated successfully!")
print("=" * 70)

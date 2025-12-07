import cv2
import numpy as np
import os
from TrainerComponent.data_utils import load_samples, CHARACTERS
from collections import Counter

def inspect_training_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
    
    # Load ALL samples
    samples = load_samples(
        os.path.join(dataset_dir, "Training", "training_labels.csv"),
        os.path.join(dataset_dir, "Training", "training_words")
    )
    
    print(f"Total samples: {len(samples)}")
    
    # Check sample distribution
    label_lengths = []
    unique_labels = set()
    label_counter = Counter()
    
    # Check first 5 and last 5 samples
    check_indices = list(range(min(5, len(samples)))) + list(range(-5, 0))
    check_indices = [i for i in check_indices if 0 <= i < len(samples)]
    
    for idx in check_indices:
        img_path, label = samples[idx]
        # Load and check image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Cannot read image: {img_path}")
            continue
            
        # Check image quality
        img_mean = np.mean(img)
        img_std = np.std(img)
        
        print(f"Sample {idx}:")
        print(f"  Image: {img_path}")
        print(f"  Label: '{label}' (length: {len(label)})")
        print(f"  Image shape: {img.shape}, mean: {img_mean:.1f}, std: {img_std:.1f}")
        
        # Show characters in label
        valid_chars = [c for c in label if c in CHARACTERS]
        invalid_chars = [c for c in label if c not in CHARACTERS]
        if invalid_chars:
            print(f"  WARNING: Invalid characters in label: {invalid_chars}")
        print()
    
    # Process ALL samples
    print("\n🔍 Processing all samples...")
    for img_path, label in samples:
        label_lengths.append(len(label))
        unique_labels.add(label)
        label_counter[label] += 1
    
    # Statistics
    print("\n📊 Data Statistics (ALL SAMPLES):")
    print(f"Total samples: {len(samples)}")
    print(f"Average label length: {np.mean(label_lengths):.1f}")
    print(f"Min label length: {min(label_lengths)}")
    print(f"Max label length: {max(label_lengths)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    
    # Show all unique labels
    print(f"\n📋 All unique labels ({len(unique_labels)} total):")
    for label in sorted(unique_labels):
        print(f"  '{label}'")
    
    # Show label distribution
    print(f"\n📊 Label distribution (top 20 most common):")
    for label, count in label_counter.most_common(20):
        print(f"  '{label}': {count} samples ({count/len(samples)*100:.1f}%)")
    
    # Check character distribution
    all_chars = ''.join([label for _, label in samples])
    char_counts = {}
    for char in CHARACTERS:
        count = all_chars.count(char)
        if count > 0:
            char_counts[char] = count
    
    print(f"\n📊 Character distribution (all characters with count > 0):")
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars:
        print(f"  '{char}': {count}")
    
    # Check for any patterns
    print(f"\n🔍 Checking for data patterns...")
    
    # Check if labels match your medical terms list
    medical_terms = [
        "Aceta", "Ace", "Alatrol", "Amodis", "Atrizin", "Axodin", "Azithrocin",
        "Azyth", "Az", "Bacaid", "Backtone", "Baclofen", "Baclon", "Bacmax",
        "Beklo", "Bicozin", "Canazole", "Candinil", "Cetisoft", "Conaz", "Dancel",
        "Denixil", "Diflu", "Dinafex", "Disopan", "Esonix", "Esoral", "Etizin",
        "Exium", "Fenadin", "Fexofast", "Fexo", "Filmet", "Fixal", "Flamyd",
        "Flexibac", "Flexilax", "Flugal", "Ketocon", "Ketoral", "Ketotab",
        "Ketozol", "Leptic", "Lucan-R", "Lumona", "M-Kast", "Maxima", "Maxpro",
        "Metro", "Metsina", "Monas", "Montair", "Montene", "Montex", "Napa Extend",
        "Napa", "Nexcap", "Nexum", "Nidazyl", "Nizoder", "Odmon", "Omastin",
        "Opton", "Progut", "Provair", "Renova", "Rhinil", "Ritch", "Rivotril",
        "Romycin", "Rozith", "Sergel", "Tamen", "Telfast", "Tridosil", "Trilock",
        "Vifas", "Zithrin"
    ]
    
    missing_terms = [term for term in medical_terms if term not in unique_labels]
    present_terms = [term for term in medical_terms if term in unique_labels]
    
    print(f"\n📋 Medical terms in training data: {len(present_terms)}/{len(medical_terms)}")
    print(f"Present: {present_terms}")
    print(f"\n❌ Missing from training data: {len(missing_terms)} terms")
    if missing_terms:
        print("First 10 missing terms:", missing_terms[:10])
        if len(missing_terms) > 10:
            print(f"... and {len(missing_terms)-10} more")

if __name__ == "__main__":
    inspect_training_data()
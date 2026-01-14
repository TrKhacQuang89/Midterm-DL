import os
import shutil
from tqdm import tqdm

def simplify_dataset(src_root, dst_root):
    # Mapping old classes to 10 new simplified classes
    mapping = {
        'Resistor': ['Resistor'],
        'Capacitor': ['Capacitor-10mf', 'Capacitor-470mf', 'Film-Capacitor'],
        'LED': ['LED-Light'],
        'IC': ['IC-Chip', 'IC-Base-14-Pin', 'IC-Base-28-Pin'],
        'Diode': ['Diode', 'Zener-Diode'],
        'Transistor': ['BJT-Transistor', 'MOSFET', 'IGBT'],
        'Switch': ['Tact-Switch', 'Push-Switch', 'Rocker-Switch'],
        'Potentiometer': ['Taper-Potentiometer', 'Trimmer-Potentiometer'],
        'Battery': ['9-Volt-Battery', '1-5-Volt-Battery', '3-3-Volt-Battery'],
        'Display': ['7-Segment-Display', 'OLED-Display']
    }

    splits = ['train', 'valid', 'test']
    
    print(f"Simplifying dataset from {src_root} to {dst_root}...")

    for split in splits:
        src_split_dir = os.path.join(src_root, split)
        if not os.path.exists(src_split_dir):
            continue
            
        print(f"Processing {split} split...")
        
        for new_class, old_classes in mapping.items():
            dst_class_dir = os.path.join(dst_root, split, new_class)
            os.makedirs(dst_class_dir, exist_ok=True)
            
            for old_class in old_classes:
                src_class_dir = os.path.join(src_split_dir, old_class)
                if not os.path.exists(src_class_dir):
                    continue
                
                files = os.listdir(src_class_dir)
                for f in files:
                    src_file = os.path.join(src_class_dir, f)
                    # To avoid name collision, prefix with old class name
                    dst_file = os.path.join(dst_class_dir, f"{old_class}_{f}")
                    shutil.copy2(src_file, dst_file)

    # Summary
    print("\nDataset Summary:")
    for split in splits:
        split_path = os.path.join(dst_root, split)
        if os.path.exists(split_path):
            total_images = sum([len(os.listdir(os.path.join(split_path, c))) for c in os.listdir(split_path)])
            print(f"- {split}: {total_images} images in {len(os.listdir(split_path))} classes")

if __name__ == "__main__":
    SRC = r"e:\learn_midterm\classification_dataset"
    DST = r"e:\learn_midterm\final_classification_dataset"
    simplify_dataset(SRC, DST)
    print("\nSimplification Done!")

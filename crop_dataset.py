import os
import yaml
from PIL import Image
from tqdm import tqdm
import argparse

def prepare_classification_data(src_root, dst_root):
    """
    Crops objects from an object detection dataset (YOLO format) and saves them 
    into a classification dataset structure.
    """
    # Load data.yaml for class names
    yaml_path = os.path.join(src_root, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found at {yaml_path}")
        return
        
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"Loaded {len(class_names)} classes.")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = os.path.join(src_root, split)
        if not os.path.exists(split_dir):
            print(f"Split '{split}' not found, skipping...")
            continue
            
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Images or labels directory missing in {split_dir}, skipping...")
            continue
            
        # Create destination directories for each class within the split
        for class_name in class_names:
            os.makedirs(os.path.join(dst_root, split, class_name), exist_ok=True)
            
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing '{split}' split ({len(image_files)} images)...")
        for img_file in tqdm(image_files):
            img_path = os.path.join(images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            try:
                img = Image.open(img_path)
                w, h = img.size
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                        
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO to pixel coordinates
                    left = (x_center - width / 2) * w
                    top = (y_center - height / 2) * h
                    right = (x_center + width / 2) * w
                    bottom = (y_center + height / 2) * h
                    
                    # Ensure coordinates are within image bounds and integers
                    left = max(0, int(left))
                    top = max(0, int(top))
                    right = min(w, int(right))
                    bottom = min(h, int(bottom))
                    
                    # Skip invalid crops
                    if right <= left or bottom <= top:
                        continue
                        
                    class_name = class_names[cls_id]
                    crop = img.crop((left, top, right, bottom))
                    
                    # Target filename: originalName_bboxIndex.jpg
                    save_name = f"{os.path.splitext(img_file)[0]}_crop{i}.jpg"
                    save_path = os.path.join(dst_root, split, class_name, save_name)
                    
                    # Convert to RGB if necessary (e.g., if image is RGBA)
                    if crop.mode in ("RGBA", "P"):
                        crop = crop.convert("RGB")
                        
                    crop.save(save_path, quality=95)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    # Hardcoded paths based on project structure
    SRC_DIR = r"e:\learn_midterm\data\ElectroCom61 A Multiclass Dataset for Detection of Electronic Components\ElectroCom-61_v2"
    DST_DIR = r"e:\learn_midterm\classification_dataset"
    
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    
    prepare_classification_data(SRC_DIR, DST_DIR)
    print("Done!")

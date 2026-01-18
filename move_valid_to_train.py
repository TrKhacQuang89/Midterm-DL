"""
Script to move 50% of validation images to training set
"""
import os
import shutil
import random

def move_half_valid_to_train(base_path, classes):
    """
    Move 50% of validation images to training set for each class.
    
    Parameters:
    -----------
    base_path : str
        Path to the dataset directory (e.g., 'final_classification_dataset')
    classes : list
        List of class names to process
    """
    valid_path = os.path.join(base_path, 'valid')
    train_path = os.path.join(base_path, 'train')
    
    total_moved = 0
    
    for class_name in classes:
        valid_class_dir = os.path.join(valid_path, class_name)
        train_class_dir = os.path.join(train_path, class_name)
        
        if not os.path.exists(valid_class_dir):
            print(f"Warning: {valid_class_dir} does not exist. Skipping.")
            continue
            
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
            print(f"Created directory: {train_class_dir}")
        
        # Get all image files
        files = [f for f in os.listdir(valid_class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and take 50%
        random.shuffle(files)
        num_to_move = len(files) // 2
        files_to_move = files[:num_to_move]
        
        print(f"\n{class_name}:")
        print(f"  Total in valid: {len(files)}")
        print(f"  Moving to train: {num_to_move}")
        
        # Move files
        for filename in files_to_move:
            src = os.path.join(valid_class_dir, filename)
            dst = os.path.join(train_class_dir, filename)
            shutil.move(src, dst)
            total_moved += 1
        
        remaining = len(files) - num_to_move
        print(f"  Remaining in valid: {remaining}")
    
    print(f"\n{'='*50}")
    print(f"Total files moved: {total_moved}")
    print(f"{'='*50}")

if __name__ == "__main__":
    base_path = r"e:\learn_midterm\final_classification_dataset"
    classes = ["Battery", "Capacitor", "Transistor"]
    
    print("Moving 50% of validation images to training set...")
    print(f"Dataset path: {base_path}")
    print(f"Classes: {classes}")
    print("="*50)
    
    move_half_valid_to_train(base_path, classes)
    
    print("\nDone! Please re-run the training script to see the updated dataset sizes.")

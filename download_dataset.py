import kagglehub
import shutil
import os

# 1. Download dataset
print("Downloading dataset...")
src_path = kagglehub.dataset_download("faiyazabdullah/electrocom61")

# 2. Define destination (current workspace/data)
dest_path = os.path.join(os.getcwd(), "data")

# 3. Move files
print(f"Moving dataset from {src_path} to {dest_path}...")
if os.path.exists(dest_path):
    print(f"Warning: {dest_path} already exists. Deleting old data...")
    shutil.rmtree(dest_path)

shutil.move(src_path, dest_path)

print(f"Successfully moved dataset to: {dest_path}")
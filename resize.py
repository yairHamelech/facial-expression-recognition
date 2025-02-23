import os
import shutil
import random

# Paths
input_folder = "./archive"
output_folder = "data"

train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Set validation split ratio
val_ratio = 0.2

# Loop through emotion categories
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)

    if not os.path.isdir(category_path):
        continue  # Skip if not a directory

    images = os.listdir(category_path)
    random.shuffle(images)  # Shuffle images randomly

    val_count = int(len(images) * val_ratio)

    train_images = images[val_count:]  # 80%
    val_images = images[:val_count]  # 20%

    # Create category subfolders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # Move images
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))

    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

print("Dataset split completed!")

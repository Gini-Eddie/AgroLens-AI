import os
import shutil
import random

# CONFIGURATION
# Make sure these paths match your actual folder names
base_dir = "data"  # The folder that contains 'train' and 'val'
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
split_ratio = 0.2  # 20% of data goes to validation


def split_dataset():
    # 1. Check if train dir exists
    if not os.path.exists(train_dir):
        print(f"❌ Error: Could not find training folder at '{train_dir}'")
        return

    # 2. Get all crop classes (folders) in train directory
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    print(f"found {len(classes)} classes: {classes}\n")
    print("🚀 Starting data split...")

    total_moved = 0

    for crop_class in classes:
        # Define paths
        class_train_path = os.path.join(train_dir, crop_class)
        class_val_path = os.path.join(val_dir, crop_class)

        # Create the corresponding folder in 'val' if it doesn't exist
        os.makedirs(class_val_path, exist_ok=True)

        # Get all images in this class
        images = os.listdir(class_train_path)
        random.shuffle(images)  # Shuffle to ensure random selection

        # Calculate how many to move
        num_to_move = int(len(images) * split_ratio)
        images_to_move = images[:num_to_move]

        print(f"   ➡️ Moving {num_to_move} images from {crop_class}...")

        # Move the files
        for img_name in images_to_move:
            src = os.path.join(class_train_path, img_name)
            dst = os.path.join(class_val_path, img_name)
            shutil.move(src, dst)

        total_moved += num_to_move

    print("\n" + "=" * 30)
    print(f"✅ DONE! Moved {total_moved} images to '{val_dir}'")
    print("=" * 30)


if __name__ == "__main__":
    split_dataset()
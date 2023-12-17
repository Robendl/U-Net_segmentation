import os
import shutil
import numpy as np

# Define the total number of images
total_images = 3064

dir_train_imgs = "brain_tumour/train/images"
dir_train_labels = "brain_tumour/train/masks"
dir_test_imgs = "brain_tumour/test/images"
dir_test_labels = "brain_tumour/test/masks"

# Create directories if they don't exist
os.makedirs(dir_train_imgs, exist_ok=True)
os.makedirs(dir_train_labels, exist_ok=True)
os.makedirs(dir_test_imgs, exist_ok=True)
os.makedirs(dir_test_labels, exist_ok=True)

# Generate random indices for 90% train and 10% test
indices = np.arange(1, total_images + 1)
np.random.shuffle(indices)
train_size = int(total_images * 0.9)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Move images and masks based on the selected indices
for new_idx, idx in enumerate(train_indices):
    image_path = os.path.join(dir_train_imgs, str(new_idx) + ".png")
    label_path = os.path.join(dir_train_labels, str(new_idx) + ".png")
    shutil.copy(f'brain_tumour/images/{idx}.png', image_path)
    shutil.copy(f'brain_tumour/masks/{idx}.png', label_path)

for new_idx, idx in enumerate(test_indices):
    image_path = os.path.join(dir_test_imgs, str(new_idx) + ".png")
    label_path = os.path.join(dir_test_labels, str(new_idx) + ".png")
    shutil.copy(f'brain_tumour/images/{idx}.png', image_path)
    shutil.copy(f'brain_tumour/masks/{idx}.png', label_path)
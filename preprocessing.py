import os
import shutil
import random

# Source and destination directories
train_non_tumor_simclr_dir = 'data/train/non_tumor_images_simclr'
train_non_tumor_cnn_dir = 'data/train/non_tumor_images_cnn'
train_tumor_dir = 'data/train/tumor_images'
train_labels_dir = 'data/train/tumor_labels'

test_non_tumor_simclr_dir = 'data/test/non_tumor_images_simclr'
test_non_tumor_cnn_dir = 'data/test/non_tumor_images_cnn'
test_tumor_dir = 'data/test/tumor_images'
test_labels_dir = 'data/test/tumor_labels'

os.makedirs(train_non_tumor_simclr_dir, exist_ok=True)
os.makedirs(train_non_tumor_cnn_dir, exist_ok=True)
os.makedirs(train_tumor_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_non_tumor_simclr_dir, exist_ok=True)
os.makedirs(test_non_tumor_cnn_dir, exist_ok=True)
os.makedirs(test_tumor_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

non_tumor_indices = list(range(16000))
tumor_indices = list(range(1638))
random.shuffle(non_tumor_indices)
random.shuffle(tumor_indices)

train_non_tumor_simclr_ids = non_tumor_indices[:13362]
train_non_tumor_cnn_ids = non_tumor_indices[13362:14950]
train_tumor_ids = tumor_indices[:1588]

test_non_tumor_simclr_ids = non_tumor_indices[14950:15950]
test_non_tumor_cnn_ids = non_tumor_indices[15950:]
test_non_tumor_cnn_ids = non_tumor_indices[15950:]
test_tumor_ids = tumor_indices[1588:1638]

print(len(train_non_tumor_simclr_ids))
print(len(train_non_tumor_cnn_ids))
print(len(train_tumor_ids))
print(len(test_non_tumor_cnn_ids))
print(len(test_non_tumor_cnn_ids))

def copy_images(source_path, dest_path, ids):
    for new_id, id in enumerate(ids):
        source_image = "image" + str(id) + ".png"
        dest_image = "image" + str(new_id) + ".png"
        source_dir = os.path.join(source_path, source_image)
        dest_dir = os.path.join(dest_path, dest_image)

        shutil.copy(source_dir, dest_dir)


        

copy_images("data/non_tumor_images_png", train_non_tumor_simclr_dir, train_non_tumor_simclr_ids)
copy_images("data/non_tumor_images_png", train_non_tumor_cnn_dir, train_non_tumor_cnn_ids)
copy_images("data/non_tumor_images_png", test_non_tumor_simclr_dir, test_non_tumor_simclr_ids)
copy_images("data/non_tumor_images_png", test_non_tumor_cnn_dir, test_non_tumor_cnn_ids)


copy_images("data/tumor_images_png", train_tumor_dir, train_tumor_ids)
copy_images("data/labels_png", train_labels_dir, train_tumor_ids)
copy_images("data/tumor_images_png", test_tumor_dir, test_tumor_ids)
copy_images("data/labels_png", test_labels_dir, test_tumor_ids)



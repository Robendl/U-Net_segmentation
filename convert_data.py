import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil


def copy_images(source_path, dest_path, ids):
    for new_id, id in enumerate(ids):
        source_image = "image" + str(id) + ".png"
        dest_image = "image" + str(new_id) + ".png"
        source_dir = os.path.join(source_path, source_image)
        dest_dir = os.path.join(dest_path, dest_image)

        shutil.copy(source_dir, dest_dir)


def convert_data():
    random_state = 12

    # Path to the directory containing your images and labels
    images_dir = 'data/imagesTr'
    labels_dir = 'data/labelsTr'
    output_dir_non_tumor_images = 'data/non_tumor_images_png'
    output_dir_tumor_images = 'data/tumor_images_png'
    output_dir_labels = 'data/labels_png'

    os.makedirs(output_dir_labels, exist_ok=True)
    os.makedirs(output_dir_non_tumor_images, exist_ok=True)
    os.makedirs(output_dir_tumor_images, exist_ok=True)

    # List all the .nii.gz files in the directory
    file_list = [file for file in os.listdir(images_dir) if file.endswith('.nii.gz') and file.startswith('lung')]
    modified_list = [file_name[:-7] for file_name in file_list]
    len_list = len(modified_list)
    label_idx = 0
    non_tumor_idx = 0

    for idx, file in enumerate(file_list):
        print(f'\r{idx}/{len(file_list)}', end='')
        # Load the label
        label_path = os.path.join(labels_dir, file)
        label = nib.load(label_path).get_fdata()

        # Load the corresponding image
        image_path = os.path.join(images_dir, file)
        img = nib.load(image_path).get_fdata()

        labels_size = label.shape[2]
        indices = list(range(0, labels_size))

        # Convert slices with tumor labels to PNG
        for i in range(labels_size):  # Loop through the slices
            label_slice = label[:, :, i]
            ones = np.sum(label_slice == 1.0)

            slice_img = ((img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i]))) * 255.0
            slice_img = slice_img.astype(np.uint8)

            if ones == 0:
                output_file_imgs = os.path.join(output_dir_non_tumor_images, ("image" + str(non_tumor_idx) + ".png"))
                plt.imsave(output_file_imgs, slice_img)
                non_tumor_idx = non_tumor_idx + 1

            if ones > 0.005 * (label_slice.shape[0] * label_slice.shape[1]):
                # Save the slice as PNG
                output_file_imgs = os.path.join(output_dir_tumor_images, ("image" + str(label_idx) + ".png"))
                output_file_labels = os.path.join(output_dir_labels, ("image" + str(label_idx) + ".png"))
                plt.imsave(output_file_imgs, slice_img)
                plt.imsave(output_file_labels, label_slice)

                label_idx = label_idx + 1

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


    non_tumor_dir = os.listdir(output_dir_non_tumor_images)
    num_non_tumor = len([file for file in non_tumor_dir if os.path.isfile(os.path.join(output_dir_non_tumor_images, file))])
    # print("len files", num_non_tumor)

    tumor_dir = os.listdir(output_dir_tumor_images)
    num_tumor = len([file for file in tumor_dir if os.path.isfile(os.path.join(output_dir_tumor_images, file))])
    # print("len files", num_tumor)
    # print((num_non_tumor-num_tumor-50))
    # print((num_tumor-50))
    # exit()

    non_tumor_indices = list(range(num_non_tumor))
    tumor_indices = list(range(num_tumor))
    random.shuffle(non_tumor_indices)
    random.shuffle(tumor_indices)

    train_non_tumor_simclr_ids = non_tumor_indices[:(num_non_tumor-num_tumor-1000)]
    train_non_tumor_cnn_ids = non_tumor_indices[(num_non_tumor-num_tumor-1000):(num_non_tumor-1050)]
    train_tumor_ids = tumor_indices[:(num_tumor-50)]

    test_non_tumor_simclr_ids = non_tumor_indices[14950:15950]
    test_non_tumor_cnn_ids = non_tumor_indices[15950:]
    test_tumor_ids = tumor_indices[:num_tumor]

    print(len(train_non_tumor_simclr_ids))
    print(len(train_non_tumor_cnn_ids))
    print(len(train_tumor_ids))
    print(len(test_non_tumor_cnn_ids))
    print(len(test_non_tumor_cnn_ids))

    copy_images("data/non_tumor_images_png", train_non_tumor_simclr_dir, train_non_tumor_simclr_ids)
    copy_images("data/non_tumor_images_png", train_non_tumor_cnn_dir, train_non_tumor_cnn_ids)
    copy_images("data/non_tumor_images_png", test_non_tumor_simclr_dir, test_non_tumor_simclr_ids)
    copy_images("data/non_tumor_images_png", test_non_tumor_cnn_dir, test_non_tumor_cnn_ids)

    copy_images("data/tumor_images_png", train_tumor_dir, train_tumor_ids)
    copy_images("data/labels_png", train_labels_dir, train_tumor_ids)
    copy_images("data/tumor_images_png", test_tumor_dir, test_tumor_ids)
    copy_images("data/labels_png", test_labels_dir, test_tumor_ids)


if __name__ == '__main__':
    convert_data()

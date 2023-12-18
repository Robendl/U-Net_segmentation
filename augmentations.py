import cv2
import numpy as np
import random

def perform_augmentations(image, mask):
    # Rotate
    angle = np.random.uniform(-35, 35)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

    # Horizontal Flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # Vertical Flip
    if np.random.rand() < 0.1:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)


    return image, mask


def perform_augmentations_simclr(image1, image2):
    images = []

    for image in [image1, image2]:
        augmentations = [gaussian_blurring, translation, rotate, zoom_out]

        noise_type = np.random.choice([gaussian_noise, salt_pepper_noise])
        augmentations.append(noise_type)
        random.shuffle(augmentations)

        for aug in augmentations:
            image = aug(image)

        images.append(image)
    
    return images[0], images[1]


def gaussian_blurring(image):
    if np.random.rand() < 0.5:        
        blur_idx = random.choice([num for num in range(1, 41) if num % 2 != 0])
        image = cv2.GaussianBlur(image, (blur_idx, blur_idx), 0)

    return image

def gaussian_noise(image):
    if np.random.rand() < 0.5:
        std_dev = np.random.randint(0,15)
        mean = np.random.randint(0,10)

        row, col, ch = image.shape
        gauss = np.random.normal(mean, std_dev, (row, col, ch)).astype(np.uint8)
        gauss = gauss.reshape(row, col, ch)

        image = cv2.add(image, gauss)
    
    return image

def translation(image):
    if np.random.rand() < 0.5:
        rows, cols = image.shape[:2]
        x_translation = np.random.randint(20,100)
        y_translation = np.random.randint(20,100)

        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        image = cv2.warpAffine(image, M, (cols, rows))

    return image

def rotate(image):
    if np.random.rand() < 0.5:
        height, width = image.shape[:2]
        angle = random.randint(0, 360)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return image

def zoom_out(image):
    if np.random.rand() < 0.5:
        height, width = image.shape[:2]

        zoom_factor = random.uniform(0.5, 0.8)
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        top = random.randint(0, height - new_height)
        left = random.randint(0, width - new_width)

        roi = image[top: top + new_height, left: left + new_width]
        image = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA)

    return image

def salt_pepper_noise(image):
    if np.random.rand() < 0.5:
        height, width = image.shape[:2]

        noise_probability = np.random.uniform(0.05, 0.15)
        salt_vs_pepper = 0.5  # Equal amounts of salt and pepper noise

        # Salt mode
        num_salt = np.ceil(noise_probability * image.size * salt_vs_pepper)
        salt_coordinates = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        image[salt_coordinates[0], salt_coordinates[1]] = 1

        # Pepper mode
        num_pepper = np.ceil(noise_probability * image.size * (1.0 - salt_vs_pepper))
        pepper_coordinates = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        image[pepper_coordinates[0], pepper_coordinates[1]] = 0

    return image
import cv2
import numpy as np

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

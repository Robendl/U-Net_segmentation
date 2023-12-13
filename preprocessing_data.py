import cv2
import numpy as np

# Load the image (replace 'your_image.png' with the actual filename)
image = cv2.imread('data/train/tumor_images/image100.png', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to create a binary mask
_, thresholded = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

# Use morphological operations to refine the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

# Apply the mask to extract the lung area
result = cv2.bitwise_and(image, image, mask=mask)

# Visualize the results
cv2.imshow('Original Image', image)
cv2.imshow('Lung Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

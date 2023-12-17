import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_mask():
    # Load the image (replace 'your_image.png' with the actual filename)
    image = cv2.imread('data/train/tumor_images/image100.png', cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to create a binary mask
    _, thresholded = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

    # Use morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to extract the lung area
    result = cv2.bitwise_and(image, image, mask=mask)

    # Visualize the results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Lung Mask', mask)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    remove_columns = []

    for i in range(0, mask.shape[1] - 1):
        threshold = mask.shape[0] * 0.9
        print(np.sum(mask[i]) == 0.0)
        if 0.9 * np.sum(mask[i]) == 0.0 > 116000:
            remove_columns.append(i)
        
    print(remove_columns)
    exit()

    # Remove columns 1 and 2 (indexing starts from 0)
    new_arr = np.delete(mask, remove_columns, axis=1)

    # Display the NumPy array as an image using Matplotlib
    plt.imshow(new_arr, cmap='gray')  # 'cmap' sets the colormap (use 'gray' for grayscale)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    # Display the NumPy array as an image using Matplotlib
    plt.imshow(mask, cmap='gray')  # 'cmap' sets the colormap (use 'gray' for grayscale)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()



if __name__ == '__main__':
    create_mask()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as r


def crop_and_resize(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to create a binary mask
    _, thresholded = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

    # Use morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    height, width = np.shape(mask)
    ranges = [range(width-1), range(width-1, 0, -1), range(height-1), range(height-1, 0, -1)]
    sizes = [width, width, height, height]
    edges = []
    for j in range(len(sizes)):
        loop_range = ranges[j]
        size = sizes[j]
        for i in loop_range:
            if (j < 2 and np.sum(mask[i, :]) > 0.05) or np.sum(mask[:, i] > 0.05):
                edges.append(i)
                break

    cropped_image = image[edges[0]:edges[1], edges[2]:edges[3]]
    resized_image = cv2.resize(cropped_image, (width, height))

    cropped_label = label[edges[0]:edges[1], edges[2]:edges[3]]
    resized_label = cv2.resize(cropped_label, (width, height))

    return resized_image, resized_label
    # cv2.imshow("Original", image)
    # cv2.imshow("Cropped", cropped_image)
    # cv2.imshow("Resized", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image[edges[0], :] = 255
    # image[edges[1], :] = 255
    # image[:, edges[2]] = 255
    # image[:, edges[3]] = 255
    #
    # mask[edges[0], :] = 255
    # mask[edges[1], :] = 255
    # mask[:, edges[2]] = 255
    # mask[:, edges[3]] = 255
    #
    # plt.imshow(mask, cmap='gray')  # 'cmap' sets the colormap (use 'gray' for grayscale)
    # plt.axis('off')  # Turn off axis labels and ticks
    # plt.show()


if __name__ == '__main__':
    num_images = 3064
    for _ in range(10):
        img_idx = r.randint(1, 3065)
        print(img_idx)
        create_mask(f'brain_tumour/images/{img_idx}.png', f'brain_tumour/masks/{img_idx}.png')
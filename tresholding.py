import cv2
import numpy as np
from skimage import io, color
from sklearn.metrics import f1_score
from scipy.spatial.distance import dice
from skimage.filters import threshold_multiotsu, threshold_otsu
import matplotlib
import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred):
    return 1 - dice(y_true, y_pred)


def f1_score(predicted, mask):
    TP = np.sum(predicted * mask)  # true positives
    FP = np.sum(predicted) - TP  # false positives
    FN = np.sum(mask) - TP  # false negatives
    precision = TP / (TP + FP + 1e-7)  # precision
    recall = TP / (TP + FN + 1e-7)  # recall
    F1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # F1 score
    return F1.item()  # convert tensor to scalar value


def iou_score(prediction, target, smooth=1e-5):
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def otsus_thresholding(image_path):
    image = io.imread(image_path)
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
            
    # Normalize the image and apply threshold
    image_normalized = image / 255.0
       
    thresholds = threshold_multiotsu(image_normalized)
    optimal_threshold = thresholds[len(thresholds) // 2] #select the second threshold
    binary_mask = (image_normalized > optimal_threshold).astype(np.uint8)
    return binary_mask


def threshold():
    # Load the training images and their corresponding masks
    image_paths = []
    mask_paths = []
    base_mask_path = "brain_tumour/train/masks/"
    base_image_path = "brain_tumour/train/images/"

    for i in range(3064):
        image_paths.append(f"{base_image_path}{i+1}.png")
        mask_paths.append(f"{base_mask_path}{i+1}.png")    

    # Create empty lists to store the evaluation scores
    dice_scores = []
    f1_scores = []
    iou_scores = []

    # Loop through images and masks to apply threshold and evaluate
    for image_path, mask_path in zip(image_paths, mask_paths):
        binary_mask = otsus_thresholding(image_path)
        
        # Save the binary mask as an image
        binary_mask_path = mask_path.replace('masks', 'predicted_masks')  # Change directory accordingly
        io.imsave(binary_mask_path, binary_mask * 255)  # Save as uint8 image
        
        # Load the original ground truth mask
        ground_truth_mask = io.imread(mask_path, as_gray=True)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
        
        # Calculate Dice score and F1 score
        dice_score = dice_coefficient(ground_truth_mask.flatten(), binary_mask.flatten())
        f1 = f1_score(ground_truth_mask.flatten(), binary_mask.flatten()) 
        iou = iou_score(ground_truth_mask.flatten(), binary_mask.flatten())
        
        dice_scores.append(dice_score)
        f1_scores.append(f1)
        iou_scores.append(iou)

    # Print out the average Dice and F1 scores across all images
    print(f'Average Dice Score: {np.mean(dice_scores):.4f}')
    print(f'Average F1 Score: {np.mean(f1_scores):.4f}')
    print(f'Average IoU Score: {np.mean(iou_scores):.4f}')

if __name__ == '__main__':
    threshold()
    image_path = "brain_tumour/train/images/451.png"
    image = io.imread(image_path)
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
            
    # Normalize the image and apply threshold
    image_normalized = image / 255.0
    
    ## multi_otsu plot
    # thresholds = threshold_multiotsu(image_normalized)
    # optimal_threshold = thresholds[len(thresholds) // 2]
    # binary_mask = (image_normalized > optimal_threshold).astype(np.uint8)
    # thresholds = threshold_multiotsu(image)

    # # Using the threshold values, we generate the three regions.
    # regions = np.digitize(image, bins=thresholds)

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title('Original')
    # ax[0].axis('off')

    # # Plotting the histogram and the two thresholds obtained from
    # # multi-Otsu.
    # ax[1].hist(image.ravel(), bins=255)
    # ax[1].set_title('Histogram')
    # for thresh in thresholds:
    #     ax[1].axvline(thresh, color='r')

    # # Plotting the Multi Otsu result.
    # ax[2].imshow(regions, cmap='jet')
    # ax[2].set_title('Multi-Otsu result')
    # ax[2].axis('off')

    # plt.subplots_adjust()

    # plt.show()

    ## otsu plot
    # thresh = threshold_otsu(image)
    # binary = image > thresh

    # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 3, 1)
    # ax[1] = plt.subplot(1, 3, 2)
    # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].set_title('Original')
    # ax[0].axis('off')

    # ax[1].hist(image.ravel(), bins=256)
    # ax[1].set_title('Histogram')
    # ax[1].axvline(thresh, color='r')

    # ax[2].imshow(binary, cmap=plt.cm.gray)
    # ax[2].set_title('Thresholded')
    # ax[2].axis('off')

    # plt.show()
    
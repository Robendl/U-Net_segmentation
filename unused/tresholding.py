import cv2
import numpy as np
from skimage import io, color
from sklearn.metrics import f1_score
from scipy.spatial.distance import dice
from skimage.filters import threshold_multiotsu, threshold_otsu
import matplotlib.pyplot as plt
from augmentations import *

TRAIN_TEST = "test"
AUGMENT = 0

def add_value_labels(ax, containers):
    for container in containers:  # Loop over BarContainers
        for bar in container:  # Each container contains a list of bars
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


def load_image(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #Load label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 

    image = cv2.resize(image, (512, 512))
    label = cv2.resize(label, (512, 512))

    image, label = perform_augmentations(image, label)

    image = image.astype(np.float32) / 255.0 
    label = label.astype(np.float32) / 255.0  
    return image, label


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


def otsus_thresholding(image_path, mask_path): 
    if AUGMENT:
        image, ground_truth_mask = load_image(image_path, mask_path)
    else: 
        image = io.imread(image_path)
        ground_truth_mask = io.imread(mask_path, as_gray=True)
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
            
    # Normalize the image and apply threshold
    image_normalized = image / 255.0
       
    thresholds = threshold_multiotsu(image_normalized)
    optimal_threshold = thresholds[len(thresholds) // 2] #select the second threshold
    binary_mask = (image_normalized > optimal_threshold).astype(np.uint8)
    return binary_mask, ground_truth_mask


def threshold():
    # Load the training images and their corresponding masks
    image_paths = []
    mask_paths = []
    base_mask_path = "brain_tumour/" + TRAIN_TEST + "/masks/"
    base_image_path = "brain_tumour/" + TRAIN_TEST + "/images/"

    # Only the test images dataset is actually used, but we wanted to see if there would be a difference for the train images of the dataset
    amount_of_images = 0
    if TRAIN_TEST == "train":
        amount_of_images = 2755
    else:
        amount_of_images = 307

    #for i in range(amount_of_images): #find al paths to the images
    #    image_paths.append(f"{base_image_path}{i}.png")
    #    mask_paths.append(f"{base_mask_path}{i}.png")   
    image_paths.append(f"{base_image_path}{250}.png")
    mask_paths.append(f"{base_mask_path}{250}.png")  

    # Create empty lists to store the evaluation scores
    f1_scores = []
    iou_scores = []

    # Loop through images and masks to apply threshold and evaluate
    for image_path, mask_path in zip(image_paths, mask_paths):
        binary_mask, ground_truth_mask = otsus_thresholding(image_path, mask_path)
        
        # Save the binary mask as an image
        binary_mask_path = mask_path.replace('masks', 'predicted_masks')  # Change directory accordingly
        io.imsave(binary_mask_path, binary_mask * 255)  # Save as uint8 image
        
        # Load the original ground truth mask
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
        
        # Calculate IoU score and F1 score
        f1 = f1_score(ground_truth_mask.flatten(), binary_mask.flatten()) 
        iou = iou_score(ground_truth_mask.flatten(), binary_mask.flatten())
        
        f1_scores.append(f1)
        iou_scores.append(iou)


    # Calculate the means and standard deviations
    f1_mean = np.mean(f1_scores)
    iou_mean = np.mean(iou_scores)

    f1_std = np.std(f1_scores)
    iou_std = np.std(iou_scores)

    # Print out the average IoU and F1 scores across all images
    print(f'Average F1 Score: {f1_mean:.4f}, Standard Deviation: {f1_std:.4f}')
    print(f'Average IoU Score: {iou_mean:.4f}, Standard Deviation: {iou_std:.4f}')

    # # Set the positions and width for the bars
    # positions = np.arange(2)
    # width = 0.5  # the width of the bars

    # # Plotting the mean scores with standard deviation as error bars
    # fig, ax = plt.subplots()
    # bar1 = ax.bar(positions[0], f1_mean, width, yerr=f1_std, label='F1')
    # bar2 = ax.bar(positions[1], iou_mean, width, yerr=iou_std, label='IoU')

    # # Adding labels and title
    # ax.set_ylabel('Scores')
    # ax.set_title('Mean and Standard Deviation of Evaluation Metrics')
    # ax.set_xticks(positions)
    # ax.set_xticklabels(('F1', 'IoU'))
    # ax.legend()    

    # add_value_labels(ax, [bar1, bar2])

    # # Show the plot
    # plt.show()


if __name__ == '__main__':
    threshold()
    
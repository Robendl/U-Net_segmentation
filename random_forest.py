import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future, io, color
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
from functools import partial
from augmentations import *

TRAIN_TEST = "test"

#load image with augmentations
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

#calculate f1 score
def f1_score(predicted, mask):
    TP = np.sum(predicted * mask)  # true positives
    FP = np.sum(predicted) - TP  # false positives
    FN = np.sum(mask) - TP  # false negatives
    precision = TP / (TP + FP + 1e-7)  # precision
    recall = TP / (TP + FN + 1e-7)  # recall
    F1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # F1 score
    return F1.item()  # convert tensor to scalar value

#calculate iou score
def iou_score(prediction, target, smooth=1e-5):
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

#perform RF classification
def random_forest(x_train, y_train, x_test, y_test):     
    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    # train the model
    rf.fit(x_train, y_train)

    # Predict on the test set
    y_pred_flat = rf.predict(x_test)

    # Reshape the predictions back to the original image shape
    y_pred = y_pred_flat.reshape(y_test.shape)

    return y_pred


def main():
    # Load the training images and their corresponding masks
    train_images = []
    train_masks = []
    test_images = []
    test_masks = []

    #train-test split is already done in convert_data_brain.py
    #load in train images
    base_mask_path_train = "brain_tumour/" + "train" + "/masks/"
    base_image_path_train = "brain_tumour/" + "train" + "/images/"
    amount_of_images_train = 2755
    for i in range(amount_of_images_train): #find all the images
        image = io.imread(f"{base_image_path_train}{i}.png")
        ground_truth_mask = io.imread(f"{base_mask_path_train}{i}.png", as_gray=True)
        
        # convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        image = resize(image, (512, 512), anti_aliasing=True)
        ground_truth_mask = resize(ground_truth_mask, (512, 512), anti_aliasing=True)
                
        # normalize the image
        image_normalized = image / 255.0
        train_images.append(image_normalized)
        train_masks.append(ground_truth_mask)
    
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    print("Train images done")
    
    #load in test images
    base_mask_path_test = "brain_tumour/" + "test" + "/masks/"
    base_image_path_test = "brain_tumour/" + "test" + "/images/"
    amount_of_images_test = 307   
    for i in range(amount_of_images_test): #find al paths to the images
        image = io.imread(f"{base_image_path_test}{i}.png")
        ground_truth_mask = io.imread(f"{base_mask_path_test}{i}.png", as_gray=True)
            
        # convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        image = resize(image, (512, 512), anti_aliasing=True)
        ground_truth_mask = resize(ground_truth_mask, (512, 512), anti_aliasing=True)
                
        # normalize the image
        image_normalized = image / 255.0
        test_images.append(image_normalized)
        test_masks.append(ground_truth_mask)

    test_images = np.array(test_images)
    test_masks = np.array(test_masks)
    print("Test images done")
    
    # flatten the data for training
    x_train_flat = train_images.reshape(train_images.shape[0], -1)  # Reshape to [n_samples, n_features]
    y_train_flat = train_masks.reshape(train_masks.shape[0], -1).ravel()  # Reshape to [n_samples * n_features]
    x_test_flat = test_images.reshape(test_images.shape[0], -1)  # Reshape to [n_samples, n_features]
    y_test_flat = test_masks.reshape(test_masks.shape[0], -1).ravel()  # Reshape to [n_samples * n_features]

    print("start RF")
    y_pred_flat = random_forest(x_train_flat, y_train_flat, x_test_flat, y_test_flat)
    # Reshape the predictions back to the original image shape
    y_pred = y_pred_flat.reshape(y_test_flat.shape)
    
    # Create empty lists to store the evaluation scores
    f1_scores = []
    iou_scores = []
        
    # # Save the binary mask as an image
    # binary_mask_path = mask_path.replace('masks', 'predicted_masks')  # Change directory accordingly
    # io.imsave(binary_mask_path, y_pred * 255)  # Save as uint8 image
       
    # # Load the original ground truth mask
    # ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
       
    # Calculate IoU score and F1 score
    f1 = f1_score(y_test_flat, y_pred_flat) 
    iou = iou_score(y_test_flat, y_pred_flat) 
        
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


if __name__ == '__main__':
    main()
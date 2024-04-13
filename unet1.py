import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import *
from network import *


class ImageDataset(Dataset):
    def __init__(self, indices, image_indices, transform=False):
        self.indices = indices
        self.transform = transform
        self.image_indices = image_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        random_img_number = self.image_indices[idx]

        #Load image
        #image_path = "brain_tumour/train/images/" + str(random_img_number) + ".png"

        image_path = "/home1/s3799492/machine-learning-lung/brain_tumour/train/images/" + str(random_img_number) + ".png" 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #Load label
        #label_path = "brain_tumour/train/masks/" + str(random_img_number) + ".png"
        label_path = "/home1/s3799492/machine-learning-lung/brain_tumour/train/masks/" + str(random_img_number) + ".png" 
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512))

        if self.transform:
            image, label = perform_augmentations(image, label)

        image = image.astype(np.float32) / 255.0 
        label = label.astype(np.float32) / 255.0  
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        return image, label


def bce_loss(y_pred, y_true):
    func = nn.BCEWithLogitsLoss()
    return func(y_pred.squeeze(), y_true.float().to('cuda'))


def dice_loss(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred).squeeze()
    y_true = y_true.float().to('cuda')

    smooth = 1e-5
    y_true = y_true.view(-1, 512, 512)
    y_pred = y_pred.view(-1, 512, 512)
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    dice = (2. * intersection + smooth) / (sum_ + smooth)
    return 1. - dice


def combined_loss(y_pred, y_true):
    return dice_loss(y_pred, y_true) + bce_loss(y_pred, y_true)


def validate(dataloader_valset, model, loss_function):
    model.eval()
    # validation steps
    with torch.no_grad():

        valid_loss = 0.0
        counter = 0

        for (images, labels) in dataloader_valset:
            images = images.permute(0, 3, 1, 2)
            output = model(images.float().to('cuda'))
                
            loss = loss_function(output, labels)
            valid_loss += loss.item()
            counter += 1
            
        valid_loss /= counter
    
    return valid_loss


def save_model(model, save_file):
    torch.save(model.state_dict(), save_file)


def load_path(model, path, freeze_layers):
    print('path:\t', path)
    state_dict = torch.load(path, map_location=torch.device('cuda:0'))


    if 'unet.up.bias' in state_dict:
        # Get the keys in the state_dict
        keys = list(state_dict.keys())
        # Find the index of 'head.0.weight' in the keys
        idx = keys.index('unet.up.bias')
        # Remove all keys that come after 'head.0.weight'
        keys_to_remove = keys[idx+1:]
        for key in keys_to_remove:
            del state_dict[key]

    print("keys", list(state_dict.keys()))          

    print("model before", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    model.load_state_dict(state_dict, strict=False)

    if freeze_layers:
        for name, param in model.named_parameters():
            #print("Parameter name:", name)
            param.requires_grad = False
            if name == "unet.up.bias":
                break


    for name, param in model.named_parameters():
        print("Parameter name:", name)
        print("Requires grad:", param.requires_grad)

    print("model after", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    return model


def train_unet(model, args, loss_function=combined_loss, learning_rate=0.0001, batch_size=8, num_epochs=30):
    print("FOLD\t", str(args.fold))
    dataset_size = 551

    # Define the number of folds
    n_splits = 5  # You can adjust this as needed

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True)

    # Initialize a list to store your fold indices
    fold_indices = []

    image_indices = list(range(0,dataset_size))
    best_valid_loss = np.Inf

    # Split the dataset into K folds
    for train_image_indices, valid_image_indices in kf.split(image_indices):
        print("indices\t", len(train_image_indices), len(valid_image_indices))
        train_indices = list(range(0,len(train_image_indices)))
        valid_indices = list(range(0,len(valid_image_indices)))

        train_dataset = ImageDataset(train_indices, train_image_indices, True)
        dataloader_trainset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = ImageDataset(valid_indices, valid_image_indices)
        dataloader_valset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-6)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_trainset), eta_min=0, last_epoch=-1)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            batch_counter = 1

            for (images, labels) in dataloader_trainset:
                optimizer.zero_grad()

                images = images.permute(0, 3, 1, 2)
                output = model(images.float().to('cuda'))

                loss = loss_function(output, labels)

                loss.backward()
                optimizer.step()
                batch_counter += 1
                total_loss += loss.item()

            valid_loss = validate(dataloader_valset, model, loss_function)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #save_file = "results/unet_ss.pth"
                if args.freeze_layers:
                    save_file = "/home1/s3799492/machine-learning-lung/results/unet_simclr_fold_" + str(args.fold) + "_frozen.pth"
                else:
                    save_file = "/home1/s3799492/machine-learning-lung/results/unet_fold_" + str(args.fold) + ".pth"                    
                save_model(model, save_file)

            total_loss /= batch_counter

            print("EPOCH: ", int(epoch))
            print("train loss", total_loss)
            print("valid loss", valid_loss)
        
        exit()


def parse_option():
    parser = argparse.ArgumentParser(description="Train SimCLR model")
    parser.add_argument("--fold", type=int, help="Fold to load the trained model")
    parser.add_argument("--freeze_layers", action="store_true", help="Freeze layers")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="mlp")
    model = model.cuda()
    #model = load_path(model, "./results/unet_ss.pth") #simclr_fold_1 is the best simclr path
    #model = load_path(model, "/home1/s3799492/machine-learning-lung/results/simclr_fold_1.pth", args.freeze_layers)
    train_unet(model, args)

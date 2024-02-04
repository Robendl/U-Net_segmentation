import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import *
from network import *
from sklearn.model_selection import KFold


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
        image_path = "brain_tumour/train/images/" + str(random_img_number) + ".png"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #Load label
        label_path = "brain_tumour/train/masks/" + str(random_img_number) + ".png"
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


def dice_loss(y_pred, y_true):
    smooth = 1e-5
    y_true = y_true.view(-1, 512, 512)
    y_pred = y_pred.view(-1, 512, 512)
    intersection = torch.sum(y_true * y_pred)
    sum_ = torch.sum(y_true + y_pred)
    dice = (2. * intersection + smooth) / (sum_ + smooth)
    return 1. - dice

def validate(dataloader_valset, model):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    # validation steps
    with torch.no_grad():

        valid_loss = 0.0
        counter = 0

        for (images, labels) in dataloader_valset:
            images = images.permute(0, 3, 1, 2)
            output = model(images.float().to('cuda'))
            output_sigmoid = torch.sigmoid(output)
            labels = labels.float().to('cuda')
            

            loss = dice_loss(output_sigmoid.squeeze(dim=1), labels) + bce_loss(output.squeeze(dim=1), labels)
            valid_loss += loss.item()
            counter += 1
            
        valid_loss /= counter
    
    return valid_loss

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def save_model(model, save_file):
    torch.save(model.state_dict(), save_file)

def load_path(model, path):
    print('path:\t', path)
    state_dict = torch.load(path, map_location=torch.device('cuda:0'))
    
    print("model before", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    model.load_state_dict(state_dict, strict=False)

    print("model after", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    return model

def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="mlp")
    model = model.cuda()

    #model = load_path(model, "results/simclr.pth")

    # train_indices = list(range(0,2450))
    # valid_indices = list(range(0,307))

    image_indices = list(range(0,2757))

    # train_image_indices = image_indices[0:2450]
    # valid_image_indices = image_indices[2450:2757]

    num_epochs = 30
    batch_size = 8
    learning_rate = 0.00001
    best_valid_loss = np.Inf
    bce_loss = nn.BCEWithLogitsLoss()


    for fold, (train_image_indices, valid_image_indices) in enumerate(kf.split(image_indices)):
        print("Training fold: ", fold)
        if fold==0:
            continue
        best_valid_loss = np.Inf

        #Reset all weight parameters when training with a new fold
        model.apply(weight_reset)

        # for name, param in model.named_parameters():
        #     print(name, param.data)

        train_dataset = ImageDataset(list(range(0,len(train_image_indices))), train_image_indices, True)
        dataloader_trainset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = ImageDataset(list(range(0,len(valid_image_indices))), valid_image_indices)
        dataloader_valset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_trainset), eta_min=0,
                                                                                last_epoch=-1)

        training_losses = []
        validation_losses = []
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            batch_counter = 1

            for (images, labels) in dataloader_trainset:
                optimizer.zero_grad()

                images = images.permute(0, 3, 1, 2)
                output = model(images.float().to('cuda'))
                labels = labels.float().to('cuda')
                output_sigmoid = torch.sigmoid(output)

                loss = dice_loss(output_sigmoid.squeeze(), labels) + bce_loss(output.squeeze(), labels)

                loss.backward()
                optimizer.step()
                batch_counter += 1
                total_loss += loss.item()


            valid_loss = validate(dataloader_valset, model)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_file = "results/unetmlp" + str(fold) + ".pth"
                save_model(model, save_file)

            total_loss /= batch_counter

            print("EPOCH: ", int(epoch))
            print("train loss", total_loss)
            training_losses.append(total_loss)
            print("valid loss", valid_loss)
            validation_losses.append(valid_loss)
        np.save(f'training-loss{fold}', training_losses)
        np.save(f'validation-loss{fold}', validation_losses)
        return


if __name__ == '__main__':
    main()

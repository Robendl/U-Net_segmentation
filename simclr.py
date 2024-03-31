import torch
import torch.nn as nn
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from augmentations import *
from network import *


class ImageDataset(Dataset):
    def __init__(self, indices, image_indices, transform=False):
        self.indices = indices
        self.image_indices = image_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        random_img_number = self.image_indices[idx]

        #Load image
        # image_path = "brain_tumour/train/unlab_images/" + str(random_img_number) + ".png"
        image_path = "/home1/s3799492/machine-learning-lung/brain_tumour/train/images/" + str(random_img_number) + ".png" 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (512, 512))

        image1, image2 = perform_augmentations_simclr(image, image)

        image1 = image1.astype(np.float32) / 255.0 
        image1 = torch.from_numpy(image1)

        image2 = image2.astype(np.float32) / 255.0 
        image2 = torch.from_numpy(image2)
 
        return image1, image2

def contrastive_loss(zis, zjs, batch_size, temperature):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    zijs = torch.cat((zis, zjs), dim=0)
    
    results = torch.zeros((2*batch_size, 2*batch_size)).to('cuda')
    labels = torch.zeros(2 * batch_size).to(device='cuda', dtype=torch.int64)

    for i in range(zijs.shape[0]):
        for j in range(zijs.shape[0]):
            # Calculate dot product similarity between each pair of vectors
            results[i, j] = torch.dot(zijs[i], zijs[j])

    #For filtering out positives
    diag = np.eye(2 * batch_size)
    pos1 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    pos2 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    mask = diag + pos1 + pos2
    mask = np.logical_not(mask).astype(int)
    mask = torch.from_numpy(mask)
    boolean_mask = mask != 0
    boolean_mask = boolean_mask.to('cuda')

    l_pos = torch.diag(results, batch_size)
    r_pos = torch.diag(results, -batch_size)

    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    negatives = results[boolean_mask].view(2 * batch_size, -1)


    logits = torch.cat((positives, negatives), dim=1)
    logits = logits / temperature

    loss = criterion(logits, labels) 
    return loss / (2 * batch_size)

def validate(dataloader_valset, model, batch_size):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    # validation steps
    with torch.no_grad():

        valid_loss = 0.0
        counter = 0

        for (xis, xjs) in dataloader_valset:
            xis = xis.permute(0, 3, 1, 2)
            xjs = xjs.permute(0, 3, 1, 2)

            xis = xis.float().to('cuda')
            xjs = xjs.float().to('cuda')

            zis = model(xis)
            zjs = model(xjs)

            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)

            loss = contrastive_loss(zis, zjs, batch_size, 0.5)
            valid_loss += loss.item()
            counter += 1
            
        valid_loss /= counter
    return valid_loss

def save_model(model, save_file):
    torch.save(model.state_dict(), save_file)

def load_path(model, path):
    print('path:\t', path)
    state_dict = torch.load(path, map_location=torch.device('cuda:0'))
    
    print("model before", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    model.load_state_dict(state_dict, strict=False)

    print("model after", model.state_dict()['unet.down1.conv.0.weight'][0][0][0])
    return model

def train_simclr():
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="cls")
    model = model.cuda()

    model = load_path(model, "/home4/s3806715/machine-learning-lung-oud/results/unet_simclr.pth")
    exit()

    dataset_size = 2205
    valid_split = int(dataset_size*0.9)
    train_indices = list(range(0, valid_split))
    valid_indices = list(range(0, dataset_size - valid_split))

    image_indices = list(range(0, dataset_size))
    random.shuffle(image_indices)

    train_image_indices = image_indices[:valid_split]
    valid_image_indices = image_indices[valid_split:]

    print("indices", len(train_image_indices), len(valid_image_indices))

    num_epochs = 100
    batch_size = 8
    learning_rate = 0.0001
    best_valid_loss = np.Inf

    train_dataset = ImageDataset(train_indices, train_image_indices, True)
    dataloader_trainset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = ImageDataset(valid_indices, valid_image_indices)
    dataloader_valset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_trainset), eta_min=0,
                                                                            last_epoch=-1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_counter = 1

        for (xis, xjs) in dataloader_trainset:
            optimizer.zero_grad()

            xis = xis.permute(0, 3, 1, 2)
            xjs = xjs.permute(0, 3, 1, 2)

            xis = xis.float().to('cuda')
            xjs = xjs.float().to('cuda')

            zis = model(xis)
            zjs = model(xjs)

            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)

            loss = contrastive_loss(zis, zjs, batch_size, 0.5)

            loss.backward()
            optimizer.step()
            batch_counter += 1
            total_loss += loss.item()

        valid_loss = validate(dataloader_valset, model, batch_size)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_file = "/home1/s3799492/machine-learning-lung/results/unet_simclr.pth"
            save_model(model, save_file)

        total_loss /= batch_counter

        print("EPOCH: ", int(epoch))
        print("train loss", total_loss)
        print("valid loss", valid_loss)


if __name__ == '__main__':
    train_simclr()

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

import os
from PIL import Image

from sklearn.model_selection import train_test_split
from skimage.transform import resize

from typing import Sequence

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from data import get_dataframes
from data import TomoDataset

class ProposalNet(nn.Module):
    """ 
    A CNN intended to receive low-resolution tomogram images and determine
    if they contain the object of interest or not.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.in_shape = (32, 32, 32)
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1) 
        self.conv2 = nn.Conv3d(8, 1, 3, padding=1)
        self.linear1 = nn.Linear(512, 16)
        self.linear2 = nn.Linear(16, 1)
    
    def forward(self, x):
        if x.shape != self.in_shape:
            raise ValueError(f'Input shape should be {self.in_shape}. Actual shape is {x.shape}')
        x = self.conv1(x)                   # 1x32x32x32 -> 8x32x32x32
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2)  # 8x32x32x32 -> 8x16x16x16

        x = self.conv2(x)                   # 8x16x16x16 -> 1x16x16x16
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2)  # 1x16x16x16 -> 1x8x8x8

        x = x.view(x.size(0), -1)           # 1x8x8x8 -> 512
        
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        return x
    
    def process_label(self, label):
        return torch.tensor(
            [len(points) > 0 for points in label], 
            dtype=torch.float32
        )
    
    def train_step(
            self, 
            optimizer: optim.Optimizer, 
            train_loader,
            loss_func
        ):
        super().train()
        losses = []
        for image, label, _ in train_loader:
            # Send data to device
            image = image.to(self.device)
            label = self.process_label(label).to(self.device)
            label = label.float().view(-1, 1)

            optimizer.zero_grad()

            label_pred = self.forward(image)
            loss = loss_func(label_pred, label)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        return losses

    def validate(self, val_loader):
        super().eval()
        success_count = 0
        with torch.no_grad():
            for image, label, _ in val_loader:
                image = image.to(self.device)
                label = self.process_label(label).to(self.device)
                label = label.float().view(-1, 1)

                label_pred = self.forward(image) # logit
                predicted = (label_pred >= 0.0).float()
                success_count += (predicted == label).sum().item()
        return success_count / len(val_loader)
            
    def fit(self, train_loader, val_loader, n_epochs=10):
        """ 
        Each dataset should iterate through tuples (x, y) where x is an image of
        shape self.in_shape and y is a boolean representing if the image
        contains a target or not.
        """
        # Weight the importance of positive samples more
        pos_samples = sum(len(points) > 0 for _, points, _ in train_loader.dataset)
        pos_weight = torch.tensor([len(train_loader) / pos_samples]).to(self.device)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.SGD(self.parameters(), lr=0.001)

        train_losses = []
        val_accuracies = []
        for epoch in trange(n_epochs, desc='Training region proposal network'):
            # Train
            train_losses_epoch = self.train_step(optimizer, train_loader, loss_func)
            train_losses.append(np.mean(train_losses_epoch))
            
            # Validate
            accuracy = self.validate(val_loader)
            val_accuracies.append(accuracy)

        return train_losses, val_accuracies


if __name__ == '__main__':
    train_df, val_df = get_dataframes()
    train_set = TomoDataset(
        train_df,
        train=True
    )

    val_set = TomoDataset(
        val_df,
        train=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        collate_fn=train_set.collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        collate_fn=train_set.collate_fn
    )

    
    device = 'cpu'
    model = ProposalNet(device)
    model = model.to(device)

    model.fit(
        train_loader,
        val_loader,
        2
    )

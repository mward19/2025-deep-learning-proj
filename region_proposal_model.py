import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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
    
    def fit(self, train_loader, val_loader, n_steps=100, validate_every=20, pos_weight=1.0):
        """
        Trains the model for `n_steps` batches total.
        Runs validation every `validate_every` steps on a fixed number of samples.
        """
        self.train()
        pos_weight = torch.tensor([pos_weight]).to(self.device)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.SGD(self.parameters(), lr=0.001)

        train_losses = []
        val_accuracies = []

        loader_iter = iter(train_loader)

        for step in trange(n_steps, desc='Training'):
            try:
                image, label, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                image, label, _ = next(loader_iter)

            image = image.to(self.device)
            label = self.process_label(label).to(self.device).float().view(-1, 1)

            optimizer.zero_grad()
            label_pred = self.forward(image)
            loss = loss_func(label_pred, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (step + 1) % validate_every == 0:
                acc = self.validate(val_loader, n_items=100)
                val_accuracies.append(acc)
                print(f"[Step {step + 1}] Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")

        return train_losses, val_accuracies

    def validate(self, val_loader, n_items=30):
        self.eval()
        success_count = 0
        total = 0

        with torch.no_grad():
            for image, label, _ in val_loader:
                if total >= n_items:
                    break

                image = image.to(self.device)
                label = self.process_label(label).to(self.device).float().view(-1, 1)

                label_pred = self.forward(image)
                predicted = (label_pred >= 0.0).float()
                success_count += (predicted == label).sum().item()
                total += label.size(0)

        return success_count / total if total > 0 else 0.0



if __name__ == '__main__':
    train_df, val_df = get_dataframes()
    train_set = TomoDataset(
        train_df,
        train=True
    )
    # pos_proportion = train_set.label_balance()
    # pos_weight = (1 - pos_proportion) / pos_proportion
    # print(pos_proportion)
    # print(pos_weight)
    pos_weight = 2.0

    train_sampler = train_set.get_sampler(1 / (1+pos_weight))
    # pos_weight and sampler together mean that both classes are equally important in training

    val_set = TomoDataset(
        val_df,
        train=True
    )
    val_sampler = val_set.get_sampler(0.5)

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        collate_fn=train_set.collate_fn,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        sampler=val_sampler, # Need a sampler to get a useful idea of the accuracy
        collate_fn=train_set.collate_fn
    )

    device = 'cuda'
    model = ProposalNet(device)
    model = model.to(device)

    print('Fitting model')
    train_losses, val_accuracies = model.fit(
        train_loader,
        val_loader,
        100,
        10,
        pos_weight=pos_weight
    )
    print(train_losses)
    print(val_accuracies)

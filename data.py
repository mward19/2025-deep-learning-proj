import numpy as np
import pandas as pd
from tqdm import tqdm

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

device = 'cpu'

prefix_path = '/home/mward19/nobackup/autodelete'
all_labels = pd.read_csv(prefix_path + '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

def load_full_tomogram_array(tomo_id: str, train=True, device=device, suppress_progress=True):
    set_label = 'train' if train else 'test'
    image_dir = prefix_path + f'/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/{set_label}/{tomo_id}'

    # Sort and load images
    image_names = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, name) for name in image_names]

    # Load images as a list of PyTorch tensors
    layers = [transforms.ToTensor()(Image.open(image_path)) for image_path in tqdm(image_paths, desc='Loading tomogram', disable=suppress_progress)]
    
    # Stack into a single tensor (1, D, H, W) and move to device
    tomo_tensor = torch.stack(layers, dim=1).to(device)  # Shape: (Depth, Height, Width)

    # Expand the channel dimension
    return tomo_tensor.unsqueeze(1)

def create_point_mask(image_shape, center, radius, device):
    """
    Creates a 3D mask using PyTorch, increasing from 0 on the edge to 1 at the center.
    """
    x_center, y_center, z_center = center

    # Create a grid of coordinates
    x_range = torch.arange(image_shape[0], device=device)
    y_range = torch.arange(image_shape[1], device=device)
    z_range = torch.arange(image_shape[2], device=device)
    x_grid, y_grid, z_grid = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Calculate distance from the center
    distance_array = torch.sqrt((x_grid - x_center) ** 2 + 
                                (y_grid - y_center) ** 2 + 
                                (z_grid - z_center) ** 2)

    # Create the mask: Linearly decrease intensity with distance
    mask_array = torch.clamp(1 - (distance_array / radius), min=0)

    return mask_array

def create_mask(image_shape, motor_locs, radius, device):
    """
    Creates a combined 3D mask from multiple motor locations.
    """
    mask = torch.zeros(image_shape, device=device)  # Use specified device
    for loc in motor_locs:
        one_point_mask = create_point_mask(image_shape, loc, radius=radius, device=device)
        mask = torch.maximum(mask, one_point_mask)  # Keep everything in PyTorch

    return mask  # Shape: (D, H, W), ready for processing


def get_dataframes():
    train_df, val_df = train_test_split(all_labels, test_size=0.2, random_state=42)
    return train_df, val_df


def fm_voxels(voxel_spacing: float):
    """ 
    This function gives the size of a flagellar motor in voxels, given the image voxel spacing. 
    It will determine how much overlap each region should contain.
    """
    flagellar_motor_size = 150 # angstroms. This is a guess
    return int(flagellar_motor_size / voxel_spacing)

def tile_tomogram(image_shape, overlap: int, tile_shape):
    """ 
    Get the 'bottom left corner' position of each tile 
    in an array of shape image_shape, where each tile is of 
    shape tile_shape and overlaps in each dimension by at 
    least overlap.

    Returns list of start positions (tuples of int).
    """
    start_positions_by_dim = []
    for dim_index, (image_dim, tile_dim) in enumerate(zip(image_shape, tile_shape)):
        n_chunks = image_dim // (tile_dim - overlap)
        start_positions = np.linspace(0, image_dim - tile_dim, n_chunks, dtype=int)
        start_positions_by_dim.append(start_positions)
    return list(itertools.product(*start_positions_by_dim))

def load_tile_tomogram_array(tomo_id: str, tile_pos, tile_size, train=True, device=device, suppress_progress=True):
    set_label = 'train' if train else 'test'
    image_dir = prefix_path + f'/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/{set_label}/{tomo_id}'

    # Sort and load images
    image_names = sorted(os.listdir(image_dir))[tile_pos[0]:tile_pos[0]+tile_size[0]] # Only load the layers we need in axis 0
    image_paths = [os.path.join(image_dir, name) for name in image_names]

    # Load images as a list of PyTorch tensors
    layers = [transforms.ToTensor()(Image.open(image_path)) for image_path in tqdm(image_paths, desc='Loading tomogram', disable=suppress_progress)]
    
    # Stack into a single tensor (1, D, H, W) and move to device
    tomo_tensor = torch.stack(layers, dim=1).to(device)  # Shape: (Batch size, Depth, Height, Width)

    # Return only the desired tile. It has already been trimmed in depth 
    return tomo_tensor[
        :, # This should only have size 1
        :, 
        tile_pos[1] : tile_pos[1]+tile_size[1],
        tile_pos[2] : tile_pos[2]+tile_size[2]
    ].unsqueeze(1) # Add a channel dimension

def point_in_tile(point, tile_pos, tile_size) -> bool:
    """ Check if the point (from original array) is in the proposed tile. """
    for point_dim, tile_pos_dim, tile_size_dim in zip(point, tile_pos, tile_size):
        if point_dim  < tile_pos_dim or point_dim >= tile_pos_dim + tile_size_dim:
            return False
    return True


class TomoDataset(Dataset):
    def __init__(
            self, 
            data_df: pd.DataFrame, 
            *,
            mask_mode: bool = False, # Whether to return masks for points or just list of points in a tomogram
            tile_mode: bool = True, 
            tile_size = (64, 128, 128), # Size of tiles in original tomogram
            tile_resize = (32, 32, 32), # Size to resize tiles to when loaded
            train: bool = True, # Train or test
            transform=None
        ):
        # For convenience, maps tomo_id to list of motor points
        self.motors = {tomo_id: [] for tomo_id in all_labels['tomo_id']}
        self.voxel_spacings = dict()
        self.shapes = dict()
        self.mask_mode = mask_mode

        # Things only useful in tile mode
        self.tile_mode = tile_mode
        if self.tile_mode:
            self.tile_positions = dict()
            self.tile_size = tile_size
            self.tile_resize = tile_resize
            # What to multiply by to take original dimensions to the resized ones
            self.resize_factors = np.array([new_dim / old_dim for new_dim, old_dim in zip(self.tile_resize, self.tile_size)])
        self.train = train
        ids_seen = set()
        
        for i, row in all_labels.iterrows():
            # Each row might contain a motor. If it doesn't, just move on
            if row['Number of motors'] == 0:
                continue
            tomo_id = row['tomo_id']
            motor_loc = tuple([row[f'Motor axis {axis}'] for axis
                               in range(3)])
            self.motors[tomo_id].append(motor_loc)

            # Save other data that only needs to be seen once per tomogram data row
            if tomo_id in ids_seen:
                continue
            tomo_voxel_spacing = row['Voxel spacing']
            self.voxel_spacings[tomo_id] = tomo_voxel_spacing
            tomo_shape = tuple([row[f'Array shape (axis {i})'] for i in range(3)])
            self.shapes[tomo_id] = tomo_shape
            ids_seen.add(tomo_id)

            if self.tile_mode:
                self.tile_positions[tomo_id] = tile_tomogram(
                    tomo_shape, 
                    fm_voxels(tomo_voxel_spacing), 
                    self.tile_size
                )
            
        self.tomo_ids = list(data_df['tomo_id'])
        
    def num_tiles(self, tomo_index):
        if not self.tile_mode:
            raise Exception('You must be in tile mode to access the number of tiles in a tomogram.')
        tomo_id = self.tomo_ids[tomo_index]
        return len(self.tile_positions[tomo_id])

    def get_tile_position(self, tomo_index, tile_index):
        """ Get tile position and actual size from tomogram index and tile index, as in __getitem__. """
        assert self.tile_mode
        tomo_id = self.tomo_ids[tomo_index]
        tile_pos = self.tile_positions[tomo_id][tile_index]
        return tile_pos, self.tile_size
    
    
    def __len__(self):
        return len(self.tomo_ids)
    
    def point_to_tile_loc():
        pass # TODO
    def tile_loc_to_point():
        pass # TODO
    
    def load_tile(self, tomo_index, tile_index):
        tomo_id = self.tomo_ids[tomo_index]
        voxel_spacing = self.voxel_spacings[tomo_id]
        tile_pos = self.tile_positions[tomo_id][tile_index]
        tile_array = load_tile_tomogram_array(
            tomo_id,
            tile_pos,
            self.tile_size,
            self.train
        )
        # Load all motor positions in this tile, relative to the tile
        points = [np.array(p) - np.array(tile_pos) 
                    for p in self.motors[tomo_id] 
                    if point_in_tile(p, tile_pos, self.tile_size)]

        # Resize the tile, and scale points appropriately
        tile_array_resized = F.interpolate(tile_array, size=self.tile_resize, mode='area')
        points_resized = [(p * self.resize_factors).astype(int) for p in points]

        # Scale voxel sizing by average factor (this is kind of a dumb assumption)
        voxel_spacing_resized = voxel_spacing * np.mean(self.resize_factors)
        if self.mask_mode:
            mask = create_mask(self.tile_resize, points_resized, voxel_spacing_resized)
            return tile_array_resized, mask, voxel_spacing_resized
        else:
            return tile_array_resized, points_resized, voxel_spacing_resized
        
    def load_full_tomo(self, tomo_index):
        tomo_id = self.tomo_ids[tomo_index]
        image_array = load_full_tomogram_array(tomo_id, self.train)
        # TODO: make this depend on motor size
        radius = 50
        motors = self.motors[tomo_id]
        voxel_spacing = self.voxel_spacings[tomo_id]
        if self.mask_mode:
            truth_mask = create_mask(image_array.shape, motors, radius, device)
            return image_array, truth_mask, voxel_spacing
        else:
            return image_array, motors, voxel_spacing

    def __getitem__(self, indices):
        """ 
        If in tile mode (self.tile_mode), indices is a tuple of the tomogram's index and the tile index.
        Otherwise it is just an integer representing the index of the tomogram.
        """
        if self.tile_mode:
            if isinstance(indices, int):
                raise Exception('In tile mode, loading data requires two indices -- one for the tomogram, and one for the tile.')
            tomo_index, tile_index = indices
            return self.load_tile(tomo_index, tile_index)
        else:
            index = indices
            return self.load_full_tomo(index)
    
    def collate_fn(self, batch):
        """ DataLoaders will get very mad at this class with batch sizes bigger
        than 1 since the labels are lists. This should be used as the collate_fn
        in the DataLoader initialization. """
        images, labels, spacings = zip(*batch)
        images = torch.stack(images)
        return images, labels, spacings
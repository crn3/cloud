from cloud_dataset import CloudDataset

import random
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import pandas as pd
from torch.utils.data import ConcatDataset

from pathlib import Path

def get_dataloaders(batch_size=16, train_split=0.7, num_workers=2):

# Paths

    base_path_38 = Path(r"C:\Users\racha\Desktop\Dataset\38-Cloud_training")
    csv_path_38 = Path(r"C:\Users\racha\Desktop\Dataset\training_patches_38-cloud_nonempty.csv")

    base_path_95 = Path(r"C:\Users\racha\Desktop\Dataset\95-cloud_training_only_additional_to38-cloud")
    csv_path_95 = Path(r"C:\Users\racha\Desktop\Dataset\95-cloud_training_only_additional_to38-cloud\training_patches_95-cloud_nonempty.csv")

    # Load datasets

    data_38 = CloudDataset(
        r_dir = base_path_38 / 'train_red',
        g_dir = base_path_38 / 'train_green',
        b_dir = base_path_38 / 'train_blue',
        nir_dir = base_path_38 / 'train_nir',
        gt_dir = base_path_38 / 'train_gt',
        pytorch=True,
        include_nir=True
    )

    data_95 = CloudDataset(
        r_dir = base_path_95 / 'train_red_additional_to38cloud',
        g_dir = base_path_95 / 'train_green_additional_to38cloud',
        b_dir = base_path_95 / 'train_blue_additional_to38cloud',
        nir_dir = base_path_95 / 'train_nir_additional_to38cloud',
        gt_dir = base_path_95 / 'train_gt_additional_to38cloud',
        pytorch=True,
        include_nir=True
    )

    # Load non-empty patches from CSVs

    valid_patches_38 = set(pd.read_csv(csv_path_38)["name"].astype(str))
    valid_patches_95 = set(pd.read_csv(csv_path_95)["name"].astype(str))

    # ID empty/non-empty patches in datasets

    nonempty_indices_38 = []
    empty_indices_38 = []
    nonempty_indices_95 = []
    empty_indices_95 = []

    for i, f in enumerate(data_38.files):
        patch_name = f["red"].stem.replace("red_", "")
        if patch_name in valid_patches_38:
            nonempty_indices_38.append(i)
        else:
            empty_indices_38.append(i)

    for i, f in enumerate(data_95.files):
        patch_name = f["red"].stem.replace("red_", "")
        if patch_name in valid_patches_95:
            nonempty_indices_95.append(i)
        else:
            empty_indices_95.append(i)

    print("Cloud38 dataset size:", len(data_38))
    print("Non-empty samples:", len(nonempty_indices_38))
    print("Empty samples:", len(empty_indices_38))

    print("Cloud95 dataset size:", len(data_95))
    print("Non-empty samples:", len(nonempty_indices_95))
    print("Empty samples:", len(empty_indices_95))

    # Create one complete filtered dataset

    filtered_files_38 = [data_38.files[i] for i in nonempty_indices_38]
    filtered_files_95 = [data_95.files[i] for i in nonempty_indices_95]

    data = CloudDataset(
        r_dir = base_path_38 / 'train_red',
        g_dir = base_path_38 / 'train_green',
        b_dir = base_path_38 / 'train_blue',
        nir_dir = base_path_38 / 'train_nir',
        gt_dir = base_path_38 / 'train_gt',
        pytorch=True,
        include_nir=True
    )

    # overwrite 
    data.files = filtered_files_38 + filtered_files_95
    print("Combined filtered dataset size:", len(data))

    # Train/validation splitting 

    total = len(data)
    train_size = int(total * 0.7)
    valid_size = total - train_size

    train_ds, valid_ds = random_split(data, [train_size, valid_size])

    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2)

    valid_dl = DataLoader(
        valid_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True)

    return train_dl, valid_dl






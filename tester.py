from datasets.cloud_dataset import CloudDataset
from models.unet import UNET

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import torch.nn as nn

base_path_38 = Path(r"C:\Users\racha\OneDrive\Desktop\Dataset\38-Cloud_training")

data = CloudDataset(
    r_dir = base_path_38 / 'train_red',
    g_dir = base_path_38 / 'train_green',
    b_dir = base_path_38 / 'train_blue',
    nir_dir = base_path_38 / 'train_nir',
    gt_dir = base_path_38 / 'train_gt',
    pytorch=True,
    include_nir=True
)

# small subset for testing 
small_subset, _ = random_split(data, [32, len(data) - 32])

train_dl = DataLoader(small_subset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

unet = UNET(in_channels=4, out_channels=2) 
unet.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)

unet.train()
for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    pred = unet(xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    optimizer.step()
    print('Loss:', loss.item())
    break 

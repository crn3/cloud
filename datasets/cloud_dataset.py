from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Custom CloudDataset class
# intitialised with:
# directories of each band + ground truth mask
# flag for whether to return arrays in PyTorch (C x H x W) format
# flag for whether to include the NIR band

# based upon Maurcio Cordiero's https://medium.com/analytics-vidhya/how-to-create-a-custom-dataset-loader-in-pytorch-from-scratch-for-multi-band-satellite-images-c5924e908edf


class CloudDataset(Dataset):
    def __init__(
        self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True, include_nir=True
    ):
        super().__init__()

        # loop through files in the red folder: being used as a reference list
        # combine the other bands into a dictionary
        r_dir = Path(r_dir)

        # collect only files with expected extension (filter out directories)
        red_files = sorted([f for f in r_dir.iterdir() if f.is_file()])

        # for every red file, create this list of dicts with ALL band paths
        self.files = [
            self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in red_files
        ]

        self.pytorch = pytorch
        self.include_nir = include_nir

    # Construct paths for each band using red band file
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        # assume files have same naming structure
        name = r_file.name
        return {
            "red": r_file,
            "green": Path(g_dir) / name.replace("red", "green"),
            "blue": Path(b_dir) / name.replace("red", "blue"),
            "nir": Path(nir_dir) / name.replace("red", "nir"),
            "gt": Path(gt_dir) / name.replace("red", "gt"),
        }

    # returns the number of samples
    def __len__(self):
        return len(self.files)

    # Returns a NumPy array for each sample
    # load RGB bands, stacks them (H, W, 3)
    # Optional: load and append NIR (H, W, 4)
    # Normalise integer pixel values to be between 0 and 1
    # If PyTorch = true, dimensions reordered (C, H, W)
    def open_as_array(self, idx, invert=False):

        # read red/green/blue
        # each of these is a two-dimensional array (height x width)
        f = self.files[idx]
        r = np.array(Image.open(f["red"]))
        g = np.array(Image.open(f["green"]))
        b = np.array(Image.open(f["blue"]))

        # each is stacked into three-dimensional array (height x width x 3 channels)
        rgb = np.stack([r, g, b], axis=2)

        # optionally include the NIR channel
        if self.include_nir:
            try:
                nir = np.array(Image.open(f["nir"]))
                nir = np.expand_dims(nir, axis=2)
                arr = np.concatenate([rgb, nir], axis=2)
            except FileNotFoundError:
                # fallback to just rgb if NIR missing
                arr = rgb
        else:
            arr = rgb

        # Normalising pixel values
        # based on integer dtype (typical uint8)
        # makes all values between 0 and 1
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)

        # Reorder the dimensions for PyTorch format (C, H, W)
        if invert:
            arr = arr.transpose((2, 0, 1))
        return arr

    # Load the ground truth mask
    # Convert to binary
    # add_channel_dim: loss functions often expect class indices, don't want channel dimension
    # will need a channel dimension if mask being treated as image-like tensor
    def open_mask(self, idx, add_channel_dim=False):
        f = self.files[idx]
        # mask is greyscale: 255 = cloud, 0 = not cloud
        mask = np.array(Image.open(f["gt"]))
        # convert 255 to 1, everything else 0
        mask = np.where(mask == 255, 1, 0).astype(np.uint8)

        if add_channel_dim:
            return np.expand_dims(mask, axis=0)
            # (1 x H x W) = 3D tensor with channel dimension
        else:
            return mask  # (H x W) = 2D image

    # Creates tensors for each NumPy array
    # returns a single training sample: a tuple (x,y)
    # x = 4 channel tensor (C, H, W) RGB (+NIR)
    # y = ground truth mask (H, W)
    def __getitem__(self, idx):

        x_arr = self.open_as_array(idx, invert=self.pytorch)

        # tensor expects float for inputs
        # x = torch.tensor(x_arr, dtype=torch.float32) # input image tensor (4 x H x W)
        x = torch.from_numpy(x_arr).float()
        y_arr = self.open_mask(
            idx, add_channel_dim=False
        )  # cloud ground truth mask (H x W)

        # classification / segmentation mask should be long for loss functions
        # y = torch.tensor(y_arr, dtype=torch.int64)
        y = torch.from_numpy(y_arr).long()
        return x, y

    # Returns a PIL image for visualisation
    # extracts the RGB to convert back into normal image format
    # # ignores NIR
    def open_as_pil(self, idx):
        arr = self.open_as_array(idx, invert=False)  # (H x W x C)
        rgb = arr[..., :3]  # takes all dimensions, but drops NIR if present
        rgb = (rgb * 255).astype(np.uint8)  # de-normlalises from being between 0 and 1
        return Image.fromarray(rgb, mode="RGB")

    # method override
    # Returns string representation of dataset
    def __repr__(self):
        return f"CloudDataset with {len(self)} files (include_nir={self.include_nir})"

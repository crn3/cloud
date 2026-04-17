from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


from datasets.test_dataset import TestPatchDataset
from load_model import load_checkpoint

def predict_mask(model, x, device):
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)                  # shape: [B, 2, H, W]
        pred = torch.argmax(logits, dim=1) # shape: [B, H, W]

    # convert from tensor to numpy array for plotting
    return pred.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TestPatchDataset(
        r_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_red",
        g_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_green",
        b_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_blue",
        nir_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_nir",
        include_nir=True
    )
    
    comparisons_dir = Path("comparisons")
    comparisons_dir.mkdir(exist_ok=True)
    
    COMPARISON_SAVE_PATH = comparisons_dir/f"{time.strftime('%Y%m%d_%H%M%S')}"

    # pick one patch
    idx = 3
    x, name = dataset[idx]

    # adds new dimension at 0
    # from [channel, height, width] to [batch, channel, height, width]
    # torch expects input in batches, even if it's just one image
    x = x.unsqueeze(0)

    unet = load_checkpoint(
        model_path=r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\Unet_20260413_004720\Unet.pth",
        model_type="unet",
        device=device
    )

    unetpp = load_checkpoint(
        model_path=r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\UnetPlusPlus_20260413_150345\UnetPlusPlus.pth",
        model_type="unetplusplus",
        device=device
    )

    pred_unet = predict_mask(unet, x, device)[0] # returns numpy array [b,h,w], [0] removes batch dimension so [h,w]
    pred_unetplusplus = predict_mask(unetpp, x, device)[0] #2d arrays, each pixel is 0 no cloud or 1 cloud

    # for displaying rgb image
    rgb = x[0, :3].cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    # plt.title(f"Input\n{name}")
    plt.title(f"Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_unet, cmap="gray")
    plt.title("U-Net prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_unetplusplus, cmap="gray")
    plt.title("U-Net++ prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(COMPARISON_SAVE_PATH, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
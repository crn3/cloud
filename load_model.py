import torch
import segmentation_models_pytorch as smp

def build_model(model_type):
    if model_type == "unet":
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=4,
            classes=2
        )
    elif model_type == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=4,
            classes=2
        )
    else:
        raise ValueError("Invalid model type.")
    return model


def load_checkpoint(model_path, model_type, device):
    model = build_model(model_type)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
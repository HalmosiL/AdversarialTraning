import segmentation_models_pytorch as smp
import torch

def resnet_slice_model(model, level="Encoder"):
    if(level == "Encoder"):
        return torch.nn.Sequential(model.encoder)

def get_DeepLabv3(device, encoder_weights=None):
    model = smp.DeepLabV3(
        encoder_name='resnet34',
        encoder_depth=5,
        encoder_weights=encoder_weights,
        decoder_channels=256,
        in_channels=3,
        classes=19,
        activation=None,
        upsampling=8,
        aux_params=None
    )

    model = model.to(device)

    return model

def get_resnet18_hourglass(device, encoder_weights=None):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=19,
    )

    model = model.to(device).eval()

    return model

def load_model(path, device):
    model = get_DeepLabv3(device, encoder_weights=None)
    model.load_state_dict(torch.load(path))
    model = model.to(device)

    return model

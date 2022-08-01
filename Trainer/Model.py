import segmentation_models_pytorch as smp
import torch

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

def get_model_dummy(device):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 19, 3, stride=1, padding=1),
        torch.nn.ReLU(),
    ).to(device)


def get_resnet18_hourglass(device, encoder_weights=None):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=19,
    )

    model = model.to(device)

    return model

def get_resnet34_hourglass(device, encoder_weights=None):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=19,
    )

    model = model.to(device)

    return model

def get_resnet50_hourglass(device, encoder_weights=None):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=19,
    )

    model = model.to(device)    
    
    return model


def get_vgg16_bn_hourglass(device):
    model = smp.Unet(
        encoder_name="vgg16_bn",
        encoder_weights=None,
        in_channels=3,
        classes=19,
    )

    model = torch.nn.Sequential(
        model,
        torch.nn.ReLU()
    ).to(device)

    return model

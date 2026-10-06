"""Model factory. Add new backbones by registering a builder here."""
import torch.nn as nn
from torchvision import models


def _efficientnet_b0(pretrained):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


BACKBONES = {"efficientnet_b0": _efficientnet_b0}


def build_model(backbone="efficientnet_b0", pretrained=True):
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Available: {list(BACKBONES)}")
    return BACKBONES[backbone](pretrained)

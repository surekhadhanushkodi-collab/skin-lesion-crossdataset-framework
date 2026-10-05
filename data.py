"""Datasets and image transforms for the four colour variants."""
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .colour import ShadesOfGray

VARIANTS = ("base", "sog", "coloraug", "both")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transforms(variant, image_size=224):
    """Return (train_transform, eval_transform) for one variant.

    base     : mild augmentation, no colour normalisation
    sog      : shades-of-gray colour normalisation
    coloraug : strong colour jitter + random grayscale
    both     : sog + strong colour augmentation
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    sog = variant in ("sog", "both")
    strong = variant in ("coloraug", "both")
    pre = [transforms.Resize((image_size, image_size))]
    if sog:
        pre.append(ShadesOfGray())
    jitter = (transforms.ColorJitter(0.3, 0.3, 0.3, 0.1) if strong
              else transforms.ColorJitter(0.1, 0.1))
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
           transforms.RandomRotation(20), jitter]
    if strong:
        aug.append(transforms.RandomGrayscale(0.1))
    end = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(pre + aug + end), transforms.Compose(pre + end)


class ImgDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i], dtype=torch.float32)

"""Colour normalisation used by the 'sog' and 'both' variants."""
import numpy as np
from PIL import Image


class ShadesOfGray:
    """Shades-of-Gray colour constancy (Finlayson and Trezzi, 2004).

    Estimates the illuminant with a Minkowski norm of power p and rescales each
    channel so the estimated illuminant becomes neutral grey.
    """

    def __init__(self, p=6):
        self.p = p

    def __call__(self, img):
        a = np.asarray(img, dtype=np.float64)
        illum = (a ** self.p).reshape(-1, 3).mean(0) ** (1 / self.p)
        a = a * (illum.mean() / np.maximum(illum, 1e-6))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

import torchaudio.transforms as tat
from torch import Tensor, nn
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class Crop(AugmentationBase):
    def __init__(self, value):
        self._aug = tat.FrequencyMasking(value)

    def __call__(self, data: Tensor):
        aug = self._aug(data).float()
        return aug

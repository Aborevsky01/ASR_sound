import torchaudio.transforms as tat
from torch import Tensor, nn
import numpy as np

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Crop(AugmentationBase):
    def __init__(self, value):
        self._aug = RandomApply(tat.FrequencyMasking(value), 0.05)

    def __call__(self, data: Tensor):
        aug = self._aug(data).float()
        return aug

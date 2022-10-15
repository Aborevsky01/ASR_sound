from torch_audiomentations import AddColoredNoise
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Noise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = RandomApply(AddColoredNoise(*args, **kwargs), 0.3)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

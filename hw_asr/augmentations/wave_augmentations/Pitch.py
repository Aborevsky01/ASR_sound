from torch_audiomentations import PitchShift
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Pitch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = RandomApply(PitchShift(*args, **kwargs), 0.05)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

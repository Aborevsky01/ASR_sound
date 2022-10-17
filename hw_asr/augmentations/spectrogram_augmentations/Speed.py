import torch_audiomentations
import speechbrain.processing.speech_augmentation as spm
from librosa.effects import time_stretch
import torchaudio.transforms as tat
from torch import Tensor, nn
import numpy as np

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Speed(AugmentationBase):
    def __init__(self):
        self._aug = tat.TimeStretch(n_freq=128)

    def __call__(self, data: Tensor):
        p = np.random.random()
        if p < 0.1:
            aug = self._aug(data, np.random.uniform(1., 1.3)).float()
            res = nn.functional.pad(aug, (1, abs(aug.shape[-1] - data.shape[-1])))
        return res

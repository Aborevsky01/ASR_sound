from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, argmax_pred, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(argmax_pred, lengths, text):
            target_text = CTCCharTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text.replace('_',' ')))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCCharTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, bms_pred, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        for (pred_txt, _, _, _, _), length, target_text in zip(bms_pred, lengths, text):
            target_text = CTCCharTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_txt))
        return sum(wers) / len(wers)

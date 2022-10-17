import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {k: [dic[k] for dic in dataset_items] for k in dataset_items[0]}
    spect_len, text_enc_len = [], []
    for i, item in enumerate(dataset_items):
        spect_len.append(item['spectrogram'].shape[2])
        text_enc_len.append(item['text_encoded'].shape[1])

    result_batch['text_encoded'] = \
        pad_sequence([torch.tensor(i['text_encoded']).squeeze() for i in dataset_items], batch_first=True)
    result_batch['spectrogram'] = \
        pad_sequence([torch.tensor(i['spectrogram']).squeeze().permute(1, 0) for i in dataset_items],
                                               batch_first=True).permute(0, 2, 1)
    #result_batch['audio'] = \
    #    pad_sequence([torch.tensor(i['audio']).squeeze() for i in dataset_items], batch_first=True)
    result_batch['text_encoded_length'] = torch.tensor(text_enc_len).long()
    result_batch['spectrogram_length'] = torch.tensor(spect_len).long()
    return result_batch

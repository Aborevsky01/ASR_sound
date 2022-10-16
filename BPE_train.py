import argparse
import collections
import os
import re
from string import ascii_lowercase

from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
import sentencepiece as spm

from tqdm.notebook import tqdm


def bpe_train(config):
    bpe_params = config['BPEl']
    if bpe_params['use'] == False:
        return list(ascii_lowercase + ' ')
    full_name = bpe_params['BPE_model'] + '.model'
    if not os.path.exists(ROOT_PATH / "BPE_models" / full_name):
        text_encoder = config.get_text_encoder()
        dataloaders = get_dataloaders(config, text_encoder)
        with open(ROOT_PATH / "BPE_train" / "all.txt", 'w') as f:
            for i, batch in tqdm(enumerate(dataloaders['train']), total=len(dataloaders['train'])):
                for line in batch['text']:
                    c = re.sub("[_0-9,.:;?!\"']", "", line)
                    regex = re.compile('[^a-z ]')
                    c = regex.sub('', c)
                    f.write(c.lower() + '\n')

        spm.SentencePieceTrainer.train(input=ROOT_PATH / "BPE_train" / "all.txt",
                                       model_prefix=config['BPE_model'], model_type='bpe', vocab_size=70)
    BPE = spm.SentencePieceProcessor()
    BPE.load(ROOT_PATH / "BPE_models" / full_name)
    vocab = ['_'] + [BPE.id_to_piece(id) for id in range(BPE.get_piece_size())]
    vocab = list(set(vocab).union(set(list(ascii_lowercase))))
    return vocab


def kenlm_path():
    lm_path = 'BPE_models/lowercase.arpa'
    if not os.path.exists(ROOT_PATH / "BPE_models" / lm_path):
        with open('BPE_models/4gram_big.arpa', 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    return ROOT_PATH / "BPE_models" / lm_path

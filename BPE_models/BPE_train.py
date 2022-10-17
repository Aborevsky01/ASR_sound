import argparse
import collections
import os
import re
from string import ascii_lowercase

from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
import sentencepiece as spm

from tqdm.notebook import tqdm
import gzip
import os
import shutil
import wget


def bpe_train(config):
    bpe_params = config['BPE']
    if bpe_params['use'] == False:
        return list(ascii_lowercase + ' ')
    full_name = bpe_params['BPE_model'] + '.model'
    vocab_name = bpe_params['BPE_model'] + '.vocab'
    if not os.path.exists(ROOT_PATH / "BPE_models" / full_name):
        text_encoder = config.get_text_encoder()
        dataloaders = get_dataloaders(config, text_encoder)
        if not os.path.exists(ROOT_PATH / "BPE_models" / "{0}_all.txt".format(bpe_params['BPE_model'])):
            with open(ROOT_PATH / "BPE_models" / "{0}_all.txt".format(bpe_params['BPE_model']), 'w') as f:
                for i, batch in tqdm(enumerate(dataloaders['train']), total=len(dataloaders['train'])):
                    for line in batch['text']:
                        c = re.sub("[_0-9,.:;?!\"']", "", line)
                        regex = re.compile('[^a-z ]')
                        c = regex.sub('', c)
                        f.write(c.lower() + '\n')

        spm.SentencePieceTrainer.train(input=ROOT_PATH / "BPE_models" / "{0}_all.txt".format(bpe_params['BPE_model']),
                                       model_prefix=bpe_params['BPE_model'], model_type='bpe', vocab_size=70)
        shutil.move(full_name, str(ROOT_PATH / "BPE_models"))
        os.remove(vocab_name)
    BPE = spm.SentencePieceProcessor()
    BPE.load(str(ROOT_PATH / "BPE_models" / full_name))
    vocab = ['_'] + [BPE.id_to_piece(id) for id in range(BPE.get_piece_size())]
    vocab = list(set(vocab).union(set(list(ascii_lowercase))))
    return vocab


def kenlm_path():
    lm_path = 'lowercase.arpa'
    if not os.path.exists(ROOT_PATH / "BPE_models" / lm_path):
        lm_gzip_path = '4-gram.arpa.gz'
        if not os.path.exists(lm_gzip_path):
            print('Downloading 4-gram model.')
            lm_url = 'https://openslr.elda.org/resources/11/4-gram.arpa.gz'
            lm_gzip_path = wget.download(lm_url)
            print('Downloaded the 4-gram language model.')
        else:
            print('.arpa.gz already exists.')

        uppercase_lm_path = '4gram_upper.arpa.gz'
        if not os.path.exists(uppercase_lm_path):
            with gzip.open(lm_gzip_path, 'rb') as f_zipped:
                with open(uppercase_lm_path, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print('Unzipped the 4-gram language model.')
        else:
            print('Unzipped .arpa already exists.')

        with open(uppercase_lm_path, 'r') as f_upper:
            with open(ROOT_PATH / "BPE_models" / lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())

    return str(ROOT_PATH / "BPE_models" / lm_path)

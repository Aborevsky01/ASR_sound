from lm_scorer.models.auto import AutoLMScorer as LMScorer
import kenlm
import torch
import os
import time
import gzip
import os, shutil, wget

lm_gzip_path = '3-gram.pruned.1e-7.arpa.gz'
if not os.path.exists(lm_gzip_path):
    print('Downloading pruned 3-gram model.')
    lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
    lm_gzip_path = wget.download(lm_url)
    print('Downloaded the 3-gram language model.')
else:
    print('Pruned .arpa.gz already exists.')

uppercase_lm_path = '3-gram.pruned.1e-7.arpa'
if not os.path.exists(uppercase_lm_path):
    with gzip.open(lm_gzip_path, 'rb') as f_zipped:
        with open(uppercase_lm_path, 'wb') as f_unzipped:
            shutil.copyfileobj(f_zipped, f_unzipped)
    print('Unzipped the 3-gram language model.')
else:
    print('Unzipped .arpa already exists.')

lm_path = '3gram.arpa'
if not os.path.exists(lm_path):
    with open(uppercase_lm_path, 'r') as f_upper:
        with open(lm_path, 'w') as f_lower:
            for line in f_upper:
                f_lower.write(line.lower())
print('Converted language model file to lowercase.')


start = time.time()
text = 'BUT WITH FULL RAVISHMENT THE HOURS OF PRIME SINGING RECEIVED THEY IN THE MIDST OF LEAVES THAT EVER BORE A BURDEN TO THEIR RHYMES'
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2", batch_size=batch_size)
t = time.time() - start
print(scorer.sentence_score('but with full ravishment', reduce="mean", log=True), t)

LM = os.path.join(os.path.dirname(__file__), 'kenlm/lm', 'test.arpa')
model = kenlm.LanguageModel(LM)
print('{0}-gram model'.format(model.order))
print(model.score('but with full ravishment', bos=True, eos=True), time.time() - t)

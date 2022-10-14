from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, bpe=True):
        super().__init__(alphabet, bpe)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        decoded = []
        last_char = 0
        for ind in inds:
            if ind == 0:
                continue
            if ind != last_char:
                decoded.append(self.ind2char[ind])
            last_char = ind
        return ''.join(decoded)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        # TODO: log_probs -- infinity?
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis(self.EMPTY_TOK, 1.)]
        for proba in probs:
            new_hypos: List[Hypothesis] = []
            for pos in range(len(hypos)):
                for i in range(len(proba)):
                    if hypos[pos].text[-1] == self.ind2char[i]:
                        new_hypos.append(Hypothesis(text=hypos[pos].text,
                                                    prob=hypos[pos].prob + proba[i].item() * hypos[pos].prob))
                    else:
                        new_hypos.append(Hypothesis((hypos[pos].text + self.ind2char[i]).replace(self.EMPTY_TOK, ''),
                                                    proba[i].item() * hypos[pos].prob))
            hypos = sorted(new_hypos, key=lambda x: x.prob, reverse=True)[:beam_size]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
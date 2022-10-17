from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from hw_asr.base import BaseModel
import torch


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        x = x.transpose(2, 3)
        x = nn.LayerNorm(x.shape[-1])(x)
        return x.transpose(2, 3)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pad, dropout):
        super(ConvBlock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, 2 * pad + 1, 1, padding=pad)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, 2 * pad + 1, 1, padding=pad)
        self.dropout = dropout
        self.layer_norm1 = Normalize()
        self.layer_norm2 = Normalize()

    def forward(self, x):
        residual = x
        x = nn.ReLU6()(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.cnn1(x)
        x = nn.ReLU6()(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.cnn2(x)
        return x + residual


class RNNBlock(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, batch_first):
        super(RNNBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.ReLU6(),
            nn.GRU(input_size=input_size, hidden_size=hidden_size,
                   num_layers=1, batch_first=batch_first, bidirectional=True)
        )
        self.dropout = dropout

    def forward(self, x):
        x, _ = self.seq(x)
        x = nn.Dropout(p=self.dropout)(x)
        return x


class TrialModel(BaseModel):

    def __init__(self, fc_hidden, n_class, n_feats, dropout=0.2, **batch):
        super().__init__(n_feats, n_class, **batch)
        convs = [ConvBlock(32, 32, pad=1, dropout=dropout) for _ in range(3)]
        grus = [RNNBlock(input_size=2 * fc_hidden if i != 0 else fc_hidden,
                         hidden_size=fc_hidden, dropout=dropout, batch_first=(i == 0)) for i in range(5)]

        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.resconv = nn.Sequential(*convs)
        self.fc_1 = nn.Linear(128 * 16, fc_hidden)
        self.rnn = nn.Sequential(*grus)
        self.fc_2 = nn.Linear(fc_hidden * 2, fc_hidden)
        self.out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU6(),
            nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, **batch):
        x = self.resconv(self.cnn(spectrogram.unsqueeze(1)))
        x = x.flatten(1, 2).transpose(1, 2)
        f = self.fc_1(x)
        x = self.rnn(f)
        x = self.out(self.fc_2(x) + f)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2


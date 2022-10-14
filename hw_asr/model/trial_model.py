from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel
import torch


class FullyConnected(torch.nn.Module):
    def __init__(self, n_feature: int, n_hidden: int, dropout: float, relu_max_clip: int = 20) -> None:
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x


class TrialModel(BaseModel):
    def __init__(
            self,
            n_feats: int,
            fc_hidden: int = 1024,
            n_class: int = 28,
            dropout: float = 0.0,
            **batch
    ) -> None:
        super().__init__(n_feats, n_class, **batch)
        self.n_hidden = fc_hidden
        self.fc1 = FullyConnected(n_feats, fc_hidden, dropout)
        self.fc2 = FullyConnected(fc_hidden, fc_hidden, dropout)
        self.fc3 = FullyConnected(fc_hidden, fc_hidden * 2, dropout)
        self.bi_rnn = torch.nn.RNN(fc_hidden * 2, fc_hidden, num_layers=1, nonlinearity="relu", bidirectional=True)
        self.fc4 = FullyConnected(fc_hidden, fc_hidden // 2, dropout)
        self.out = torch.nn.Linear(fc_hidden // 2, n_class)


    def forward(self, spectrogram, **batch):
        """
            Args:
                x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
            Returns:
                Tensor: Predictor tensor of dimension (batch, time, class).
            """
        # N x C x T x F

        x = self.fc1(spectrogram.transpose(1, 2))
        # N x C x T x H
        x = self.fc2(x)
        # N x C x T x H
        x = self.fc3(x)
        # N x C x T x H
        # x = x.squeeze(1)
        # N x T x H
        x = x.transpose(0, 1)
        # T x N x H
        x, _ = self.bi_rnn(x)
        # The fifth (non-recurrent) layer takes both the forward and backward units as inputs
        x = x[:, :, : self.n_hidden] + x[:, :, self.n_hidden:]
        # T x N x H
        x = self.fc4(x)
        # T x N x H
        x = self.out(x)
        # T x N x n_class
        x = x.permute(1, 0, 2)
        # N x T x n_class
        # x = torch.nn.functional.log_softmax(x, dim=2)
        # N x T x n_class
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths

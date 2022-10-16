from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F

from hw_asr.base import BaseModel
import torch


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('rnn_shape_input', x.shape)
        # x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class TrialModel(BaseModel):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, fc_hidden, n_class, n_feats, n_cnn_layers=3, n_rnn_layers=5, stride=2, dropout=0.1, **batch):
        super().__init__(n_feats, n_class, **batch)
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=64)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats * 32, fc_hidden)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=fc_hidden if i == 0 else fc_hidden * 2,
                             hidden_size=fc_hidden, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(fc_hidden * 2, fc_hidden),  # birnn returns rnn_dim*2
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, **batch):
        x = self.cnn(spectrogram.unsqueeze(1))
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2


'''
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
        self.n_hidden = fc_hidden // 2
        self.head = FullyConnected(n_feats, fc_hidden, dropout)
        self.rnn = torch.nn.GRU(input_size=fc_hidden, hidden_size=fc_hidden // 2,
                                num_layers=3, bidirectional=True, dropout=0.3, batch_first=True)
        self.convs = nn.Sequential(
            torch.nn.ReLU6(),
            torch.nn.Conv2d(1, 4, kernel_size=(41, 11), stride=2),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(4, 8, kernel_size=(21, 11), stride=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU6()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(8 * 241, 512),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, n_class)
        )


    def forward(self, spectrogram, **batch):
        """
            Args:
                x (torch.Tensor): Tensor of dimension (batch, channel, time, feature).
            Returns:
                Tensor: Predictor tensor of dimension (batch, time, class).
            """
        # N x C x T x F
        #x = self.fc1(spectrogram.transpose(1, 2))
        print('input', spectrogram.shape)
        r = self.rnn(self.head(spectrogram.transpose(1, 2)))[0].unsqueeze(1)
        print('rnn', r.shape)
        #r = r[:, :, : self.n_hidden] + r[:, :, self.n_hidden:]
        x = self.convs(r)
        print('conv', x.shape)
        out = self.out(x.transpose(2, 1).flatten(2, 3))
        print('output', out.shape)
        # x = torch.nn.functional.log_softmax(x, dim=2)
        # N x T x n_class
        return {"logits":out}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 40) // 2 - 20
'''
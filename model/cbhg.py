import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
CBHG Model
'''

class CBHG(torch.nn.Module):
    """CBHG module to convert log Mel-filterbanks to linear spectrogram.
    This is a module of CBHG introduced
    in `Tacotron: Towards End-to-End Speech Synthesis`_.
    The CBHG converts the sequence of log Mel-filterbanks into linear spectrogram.
    .. _`Tacotron: Towards End-to-End Speech Synthesis`:
         https://arxiv.org/abs/1703.10135
    """

    def __init__(
        self,
        idim,
        odim,
        conv_bank_layers=8,
        conv_bank_chans=256,
        conv_proj_filts=5,
        conv_proj_chans=512,
        highway_layers=4,
        highway_units=512,
        gru_units=512,
    ):
        """Initialize CBHG module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            conv_bank_layers (int, optional): The number of convolution bank layers.
            conv_bank_chans (int, optional): The number of channels in convolution bank.
            conv_proj_filts (int, optional):
                Kernel size of convolutional projection layer.
            conv_proj_chans (int, optional):
                The number of channels in convolutional projection layer.
            highway_layers (int, optional): The number of highway network layers.
            highway_units (int, optional): The number of highway network units.
            gru_units (int, optional): The number of GRU units (for both directions).
        """
        super(CBHG, self).__init__()
        self.idim = idim
        self.odim = odim
        self.conv_bank_layers = conv_bank_layers
        self.conv_bank_chans = conv_bank_chans
        self.conv_proj_filts = conv_proj_filts
        self.conv_proj_chans = conv_proj_chans
        self.highway_layers = highway_layers
        self.highway_units = highway_units
        self.gru_units = gru_units

        # define 1d convolution bank
        self.conv_bank = torch.nn.ModuleList()
        for k in range(1, self.conv_bank_layers + 1):
            if k % 2 != 0:
                padding = (k - 1) // 2
            else:
                padding = ((k - 1) // 2, (k - 1) // 2 + 1)
            self.conv_bank += [
                torch.nn.Sequential(
                    torch.nn.ConstantPad1d(padding, 0.0),
                    nn.Conv1d(idim, self.conv_bank_chans, 
                              kernel_size=k, padding=0),
                    torch.nn.BatchNorm1d(self.conv_bank_chans),
                    torch.nn.ReLU(),
                )
            ]

        # define max pooling (need padding for one-side to keep same length)
        self.max_pool = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, 1), 0.0), torch.nn.MaxPool1d(2, stride=1)
        )

        # define 1d convolution projection
        self.projections = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.conv_bank_chans * self.conv_bank_layers,
                self.conv_proj_chans,
                self.conv_proj_filts,
                stride=1,
                padding=(self.conv_proj_filts - 1) // 2,
                bias=True,
            ),
            torch.nn.BatchNorm1d(self.conv_proj_chans),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.conv_proj_chans,
                self.idim,
                self.conv_proj_filts,
                stride=1,
                padding=(self.conv_proj_filts - 1) // 2,
                bias=True,
            ),
            torch.nn.BatchNorm1d(self.idim),
        )

        # define highway network
        self.highways = torch.nn.ModuleList()
        self.highways += [torch.nn.Linear(idim, self.highway_units)]
        for _ in range(self.highway_layers):
            self.highways += [HighwayNet(self.highway_units)]

        # define bidirectional GRU
        self.gru = torch.nn.GRU(
            self.highway_units,
            gru_units // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # define final projection
        self.output = torch.nn.Linear(gru_units, odim, bias=True)

    def forward(self, xs):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the padded sequences of inputs (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input sequence (B,).
        Return:
            Tensor: Batch of the padded sequence of outputs (B, Tmax, odim).
            LongTensor: Batch of lengths of each output sequence (B,).
        """

        convs = []
        for k in range(self.conv_bank_layers):
            convs += [self.conv_bank[k](xs)]
        convs = torch.cat(convs, dim=1)  # (B, #CH * #BANK, Tmax)
        convs = self.max_pool(convs)
        convs = self.projections(convs).transpose(1, 2)  # (B, Tmax, idim)
        xs = xs.transpose(1, 2) + convs
        # + 1 for dimension adjustment layer
        for i in range(self.highway_layers + 1):
            xs = self.highways[i](xs)

        xs, _ = self.gru(xs)
        xs = self.output(xs)  # (B, Tmax, odim)
        xs = xs.transpose(1, 2)
        return xs

class HighwayNet(torch.nn.Module):
    """Highway Network module.
    This is a module of Highway Network introduced in `Highway Networks`_.
    .. _`Highway Networks`: https://arxiv.org/abs/1505.00387
    """

    def __init__(self, idim):
        """Initialize Highway Network module.
        Args:
            idim (int): Dimension of the inputs.
        """
        super(HighwayNet, self).__init__()
        self.idim = idim
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(idim, idim), torch.nn.ReLU()
        )
        self.gate = torch.nn.Sequential(torch.nn.Linear(idim, idim), torch.nn.Sigmoid())

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Batch of inputs (B, ..., idim).
        Returns:
            Tensor: Batch of outputs, which are the same shape as inputs (B, ..., idim).
        """
        proj = self.projection(x)
        gate = self.gate(x)
        return proj * gate + x * (1.0 - gate)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers=6, window_size=4, n_weights=25):
        super().__init__()
        self.speaker_embedding = ECAPA_TDNN(input_size=out_dim, lin_neurons=2)
        self.speaker_linear = nn.Conv1d(2, h_dim, kernel_size=1)
        self.prenet = nn.Conv1d(in_dim, h_dim, kernel_size=1)
        self.encoder = Encoder(hidden_channels=h_dim, filter_channels=h_dim*4,
                               n_heads=4, n_layers=n_layers, kernel_size=3, p_dropout=0.1, window_size=window_size)
        self.postnet = nn.Conv1d(h_dim, out_dim, kernel_size=1)
        self.weight = nn.Parameter(torch.Tensor(n_weights))
            
    def forward(self, x, y, sid=None):
        # x : (b, w, c, t)
        # y : (b, c, t)
        # sid : (b,)
        
        x = (x * torch.softmax(self.weight[None, :, None, None], dim=1)).sum(dim=1)
        speaker = self.speaker_embedding(y.transpose(1, 2))[:, 0]
        h = self.prenet(x) + self.speaker_linear(speaker.unsqueeze(-1))
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
        loss = F.l1_loss(y_pred, y)
                                 
        data = {'loss': loss,
                'y_pred': y_pred,
                'speaker': speaker
               }
        return data
    
    def inference(self, x, speaker_embedding, sid=None):
        # x : (b, c, t)
        # speaker_embedding : (b, 2)

        x = (x * torch.softmax(self.weight[None, :, None, None], dim=1)).sum(dim=1)
        h = self.prenet(x) + self.speaker_linear(speaker_embedding.unsqueeze(-1))
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
                                 
        return y_pred
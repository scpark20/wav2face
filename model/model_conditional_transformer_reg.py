import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers=6, window_size=4):
        super().__init__()
        self.embedding = nn.Embedding(256, h_dim)
        self.prenet = nn.Conv1d(in_dim, h_dim, kernel_size=1)
        self.encoder = Encoder(hidden_channels=h_dim, filter_channels=h_dim*4,
                               n_heads=4, n_layers=n_layers, kernel_size=3, p_dropout=0.1, window_size=window_size)
        self.postnet = nn.Conv1d(h_dim, out_dim, kernel_size=1)
            
    def forward(self, x, y, sid):
        # x : (b, c, t)
        # y : (b, c, t)
        # sid : (b,)
        
        sid = self.embedding(sid).unsqueeze(-1)
        h = self.prenet(x) + sid
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
        loss = F.l1_loss(y_pred, y)
                                 
        data = {'loss': loss,
                'y_pred': y_pred,
               }
        return data
    
    def inference(self, x, sid):
        # x : (b, c, t)
        
        sid = self.embedding(sid).unsqueeze(-1)
        h = self.prenet(x) + sid
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
                                 
        return y_pred
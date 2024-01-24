import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.z_encoder = ECAPA_TDNN(input_size=out_dim, lin_neurons=z_dim*2)
        self.z_linear = nn.Linear(z_dim, h_dim)
        self.prenet = nn.Conv1d(in_dim, h_dim, kernel_size=1)
        self.encoder = Encoder(hidden_channels=h_dim, filter_channels=h_dim*4,
                               n_heads=4, n_layers=6, kernel_size=3, p_dropout=0.1, window_size=4)
        self.postnet = nn.Conv1d(h_dim, out_dim, kernel_size=1)
            
    def forward(self, x, y):
        # x : (b, c, t)
        # y : (b, c, t)
        
        # (b, c), (b, c)
        z_mean, z_logstd = self.z_encoder(y.transpose(1, 2))[:, 0].split(self.z_dim, dim=1)
        z = z_mean + torch.randn_like(z_logstd).to(z_logstd.device) * z_logstd.exp()
        
        h = self.prenet(x) + self.z_linear(z).unsqueeze(2)
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
        
        recon_loss = F.l1_loss(y_pred, y)
        kl_loss = torch.mean(-z_logstd + 0.5 * (z_logstd.exp() ** 2 + z_mean ** 2) - 0.5)
        
        data = {'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'y_pred': y_pred,
                'z_mean': z_mean
               }
        return data
    
    def inference(self, x):
        # x : (b, c, t)
        
        z = torch.randn(x.size(0), self.z_dim).to(x.device)
        h = self.prenet(x) + self.z_linear(z).unsqueeze(2)
        h_mask = torch.ones(h.size(0), 1, h.size(2)).to(h.device)
        h = self.encoder(h, h_mask)
        y_pred = self.postnet(h)
                                 
        return y_pred
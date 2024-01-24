import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_flow_unconditional import Flow1d
from .transformer import Encoder

class MelEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_layer = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.encoder = Encoder(hidden_channels=hidden_channels, filter_channels=hidden_channels*4,
                               n_heads=4, n_layers=6, kernel_size=3, p_dropout=0.1, window_size=4)
        self.out_layer = nn.Conv1d(hidden_channels, out_channels*2, kernel_size=1)

    def forward(self, x):
        x = self.in_layer(x)
        x_mask = torch.ones(x.size(0), 1, x.size(2)).to(x.device)
        x = self.encoder(x, x_mask)
        x = self.out_layer(x)
        mean, logstd = x.split(x.size(1)//2, dim=1)
        return mean, logstd

class Model(nn.Module):
    def __init__(self, channels, cond_channels, hidden_channels, n_layers):
        super().__init__()
        self.original_channels = channels
        if not channels % 2 == 0:
            channels = channels + 1
        self.channels = channels
        self.flow_layers = nn.ModuleList([Flow1d(channels, hidden_channels) for _ in range(n_layers)])
        self.inverse_init = False
        self.encoder = MelEncoder(cond_channels, hidden_channels, channels)
        
    def forward(self, x, cond):
        
        if x.size(1) < self.channels:
            x = F.pad(x, (0, 0, 0, self.channels - x.size(1)))
        
        z = x
        log_det = 0
        for flow_layer in self.flow_layers:
            z, dlog_det = flow_layer(z)
            log_det = log_det + dlog_det
            
        mean, logstd = self.encoder(cond)
        loss = self.get_loss(mean, logstd, z, log_det)
        data = {'z': z,
                'log_det': log_det,
                'mean': mean,
                'logstd': logstd,
                'loss': loss
               }
        
        return data
    
    def get_loss(self, mean, logstd, z, log_det):
        dim = z.size(1) * z.size(2)
        l1 = -logstd
        l2 = -0.5 * np.log(2*np.pi)
        l3 = -0.5 * torch.exp(-2*logstd) * (z - mean)**2
        log_likelihood = torch.sum(l1 + l2 + l3, dim=(1, 2)) + log_det
        loss = torch.mean(-log_likelihood / dim)
        return loss
    
    def inference(self, cond):
        if not self.inverse_init:
            self.inverse_init = True
            self.set_inverse()

        mean, logstd = self.encoder(cond)
        z = mean + torch.randn_like(logstd) * logstd.exp()
        x = self.inverse(z)
        x = x[:, :self.original_channels]
        return x
        
    def inverse(self, z):
        x = z
        for flow_layer in reversed(self.flow_layers):
            x = flow_layer.inverse(x)
        return x
    
    def set_inverse(self):
        for flow_layer in self.flow_layers:
            flow_layer.set_inverse()
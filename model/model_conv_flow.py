import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_flow import Flow1d

class Model(nn.Module):
    def __init__(self, channels, cond_channels, hidden_channels, n_layers):
        super().__init__()
        self.original_channels = channels
        if not channels % 2 == 0:
            channels = channels + 1
        self.channels = channels
        self.flow_layers = nn.ModuleList([Flow1d(channels, cond_channels, hidden_channels) for _ in range(n_layers)])
        self.inverse_init = False
        
    def forward(self, x, cond):
        
        if x.size(1) < self.channels:
            x = F.pad(x, (0, 0, 0, self.channels - x.size(1)))
        
        z = x
        log_det = 0
        for flow_layer in self.flow_layers:
            z, dlog_det = flow_layer(z, cond)
            log_det = log_det + dlog_det
            
        loss = self.get_loss(z, log_det)
        data = {'z': z,
                'log_det': log_det,
                'loss': loss
               }
        
        return data
    
    def get_loss(self, z, log_det):
        dim = z.size(1) * z.size(2)
        log_likelihood = torch.sum(-0.5 * (np.log(2*np.pi) + z**2), dim=(1, 2)) + log_det
        loss = torch.mean(-log_likelihood / dim)
        return loss
    
    def inference(self, cond):
        if not self.inverse_init:
            self.inverse_init = True
            self.set_inverse()

        z = torch.randn(cond.shape[0], self.channels, cond.shape[2]).to(cond.device)
        x = self.inverse(z, cond)
        x = x[:, :self.original_channels]
        return x
        
    def inverse(self, z, cond):
        x = z
        for flow_layer in reversed(self.flow_layers):
            x = flow_layer.inverse(x, cond)
        return x
    
    def set_inverse(self):
        for flow_layer in self.flow_layers:
            flow_layer.set_inverse()
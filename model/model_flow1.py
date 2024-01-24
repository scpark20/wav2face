import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .flow import ResidualCouplingBlock
from .cbhg import CBHG

class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.z_dim = out_dim if out_dim % 2 == 0 else out_dim + 1
        self.encoder = CBHG(in_dim, self.z_dim)
        self.decoder = ResidualCouplingBlock(self.z_dim, 256, 5, 1, 4, gin_channels=0)
        
    def get_loss(self, z, logdet):
        dim = z.size(1) * z.size(2)
        nll = -(torch.sum(-0.5 * (np.log(2*np.pi) + z**2), dim=[1, 2]) + logdet)
        loss = torch.mean(nll / dim)
        return loss
            
    def forward(self, x, y):
        # x : (b, c, t)
        # y : (b, c, t)
        
        if y.shape[1] != self.z_dim:
            y = F.pad(y, (0, 0, 0, 1))
        
        y_mask = torch.ones(y.shape[0], 1, 1).to(x.device)
        z, logdet = self.decoder(y, y_mask)
        flow_loss = self.get_loss(z, logdet)
        
        z_pred = self.encoder(x)
        reg_loss = F.l1_loss(z_pred, z.detach())
        
        data = {'flow_loss': flow_loss,
                'reg_loss': reg_loss,
                'total_loss': flow_loss + reg_loss}
        return data
    
    def inference(self, x):
        z = self.encoder(x)
        y_mask = torch.ones(z.shape[0], 1, 1).to(x.device)
        y = self.decoder(z, y_mask, reverse=True)
        y = y[:, :self.out_dim]
        return y
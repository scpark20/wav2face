import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .flow import ResidualCouplingBlock
from .cbhg import CBHG

class VAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = CBHG(in_dim, z_dim*2)
        self.decoder = CBHG(z_dim, in_dim)

    def forward(self, y):
        # y : (b, c, t)
        
        z_params = self.encoder(y)
        z_mean, z_logstd = z_params[:, :z_params.shape[1]//2], z_params[:, z_params.shape[1]//2:]
        z_sample = z_mean + torch.randn_like(z_logstd)*z_logstd.exp()
        y_pred = self.decoder(z_sample)
        recon_loss = F.l1_loss(y, y_pred)
        kl_loss = torch.mean(-z_logstd + 0.5 * (z_logstd.exp() ** 2 + z_mean ** 2) - 0.5)
        data = {'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'vae_loss': recon_loss + kl_loss,
                'z_sample': z_sample,
                'y_pred': y_pred,
               }
        return data
    
    def inference(self, z_sample):
        y_pred = self.decoder(z_sample)
        return y_pred
        
class Model(nn.Module):
    def __init__(self, in_dim, z_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.vae = VAE(out_dim, z_dim)
        self.encoder = CBHG(in_dim, z_dim*2)
        self.decoder = ResidualCouplingBlock(z_dim, 128, 5, 1, 4, gin_channels=0)
        
    def forward(self, x, y):
        # x : (b, c, t)
        # y : (b, c, t)
        
        vae_data = self.vae(y)
        enc = self.encoder(x)
        print(enc.shape)
        z_mean, z_logstd = enc.split(self.z_dim, dim=1)
        
        z_mask = torch.ones(x.shape[0], 1, 1).to(x.device)
        z, _ = self.decoder(vae_data['z_sample'], z_mask)
        flow_loss = torch.mean(z_logstd + 0.5 * ((z_mean - z) / z_logstd.exp()) ** 2)
        
        data = {'flow_loss': flow_loss,
                'total_loss': flow_loss + vae_data['vae_loss'],
               }
        data.update(vae_data)
        return data
    
    def inference(self, x):
        z_mean, z_logstd = self.encoder(x).split(self.z_dim, dim=1)
        z_sample = z_mean + torch.randn_like(z_logstd)*z_logstd.exp()
        z_mask = torch.ones(x.shape[0], 1, 1).to(x.device)
        z_sample = self.decoder(z_sample, z_mask, reverse=True)
        y_pred = self.vae.inference(z_sample)
        return y_pred
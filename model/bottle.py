import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Bottle(nn.Module):
    def __init__(self, K, latent_channels):
        super().__init__()
        self.K = K
        self.latent_channels = latent_channels
        self.codebook_sum = None
        self.codebook_elem = None
        self.register_buffer('codebook', torch.zeros(self.K, self.latent_channels))
        self.threshold = 1.0
        self.mu = 0.99
        self.register_buffer('init', torch.zeros(1))
        
    def _quantize(self, ze):
        # ze : (n, c)
        
        w = self.codebook
        # (n, dim)
        distance = (ze**2).sum(-1, keepdim=True) -\
                   2*ze@w.T +\
                   (w.T**2).sum(0, keepdim=True)
        # (n,), (n,)
        min_distance, zi = torch.min(distance, dim=-1)
        return zi
    
    def _dequantize(self, zi):
        # zi : (n,)
        # (n, c)
        zq = F.embedding(zi, self.codebook)
        return zq
    
    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x
    
    # Choose K vectors from the data with additional noise
    def _get_codebook_from_data(self, ze):
        # ze: (K, c)
        
        codebooks = []
        k = 0
        while True:
            codebook = ze[torch.randperm(ze.shape[0])][:self.K]
            std = 0.01 / np.sqrt(self.latent_channels)
            codebook = codebook + torch.randn_like(codebook) * std
            codebooks.append(codebook)
            k += len(codebook)
            if k >= self.K:
                break
        codebook = torch.cat(codebooks, dim=0)[:self.K]
        return codebook
    
    def _initialize(self, ze):
        # ze : (n, c)
        
        self.codebook = self._get_codebook_from_data(ze)
        self.codebook_sum = self.codebook.clone()
        self.codebook_elem = torch.ones(self.K, device=self.codebook.device)
        
    def _update(self, ze, zi):
        # ze : (n, c)
        # zi : (n,)
        
        with torch.no_grad():
            '''Calculate current centroids of the z embeddings = codebook_sum/codebook_elem'''
            # (n, K)
            zi_onehot = F.one_hot(zi, num_classes=self.K).float()
            # (K, c) = (K, n) @ (n, c)
            codebook_sum_current = zi_onehot.T @ ze
            # (K,)
            codebook_elem_current = zi_onehot.sum(0)

            '''Obtain randomly a new centroids for bins whose usage is lower than the threshold'''
            # (K, c)
            codebook_random = self._get_codebook_from_data(ze)
            
            if self.init and self.codebook_sum is None:
                self.codebook_sum = self.codebook.clone()
                self.codebook_elem = torch.ones(self.K, device=self.codebook.device)

            '''Update current centroids parameters'''
            self.codebook_sum = self.mu*self.codebook_sum + (1.-self.mu)*codebook_sum_current
            self.codebook_elem = self.mu*self.codebook_elem + (1.-self.mu)*codebook_elem_current

            '''Update centroids'''
            # (K, 1)
            usage = (self.codebook_elem.unsqueeze(1) >= self.threshold).float()
            codebook_prob = self.codebook_elem / self.codebook_elem.sum()
            entropy = -torch.sum(codebook_prob*torch.log(codebook_prob + 1e-8))
            # (K, c)
            codebook_new = self.codebook_sum / self.codebook_elem.unsqueeze(1)
            # (k, c)
            self.codebook = usage*codebook_new + (1-usage)*codebook_random
            outputs = {'usage': usage.sum(),
                       'entropy': entropy}
            return outputs
        
    def forward(self, ze, q_level=1.0):
        # ze : (b, c)
        
        if not self.init:
            self.init.data.fill_(1.)
            self._initialize(ze)
            
        # z_index : (b,)
        zi = self._quantize(ze)
        # z_quantized : (b, c)
        zq = self._dequantize(zi)
        # update codebook
        outputs = self._update(ze, zi)
        # Commitment loss
        commit_loss = F.mse_loss(ze, zq)
        # pass-through
        zq = ze + (zq - ze).detach() * q_level
        outputs.update({'commit_loss': commit_loss,
                       'zi': zi,
                       'zq': zq})
        return outputs
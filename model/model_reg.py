import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbhg import CBHG

class Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = CBHG(in_dim, out_dim)
            
    def forward(self, x, y):
        # x : (b, c, t)
        # y : (b, c, t)
        
        y_pred = self.encoder(x)
        loss = F.l1_loss(y_pred, y)
        data = {'loss': loss,
                'y_pred': y_pred,
               }
        return data
    
    def inference(self, x):
        # x : (b, c, t)
        
        y_pred = self.encoder(x)
        return y_pred

   
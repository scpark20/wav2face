import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def split(x):
    c = x.size(1) // 2
    y1 = x[:, :c]
    y2 = x[:, c:]
    
    return y1, y2
    
def merge(y1, y2):
    x = torch.cat([y1, y2], dim=1)
    
    return x

class Permutation1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1 * W[:,0]
        #W = torch.eye(c)
        self.W = nn.Parameter(W)
        
    def forward(self, x):
        batch, channel, time = x.size()

        log_det_W = torch.slogdet(self.W)[1]
        dlog_det = time * log_det_W
        y = F.conv1d(x, self.W[:, :, None])

        return y, dlog_det
        
    def set_inverse(self):
        self.W_inverse = self.W.inverse()
        
    def inverse(self, y):
        x = F.conv1d(y, self.W_inverse[:, :, None])
        
        return x
    
###
class ConditionedConv1d(nn.Module):
    def __init__(self, in_channels, cond_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.c = out_channels
        self.filter_out = nn.Conv1d(in_channels=in_channels+cond_channels, out_channels=out_channels*2, 
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)      
        self.filter_out.weight.data.normal_(0, 0.02)
        
    def forward(self, x, cond):
        # x: (B, C, T)
        # cond: (B, C, T)
        y = self.filter_out(torch.cat([x, cond], dim=1))
        y = F.tanh(y[:, :self.c]) * torch.sigmoid(y[:, self.c:])
        
        return y
    
###
class NonLinear1d(nn.Module):
    def __init__(self, channels, cond_channels, hidden_channels):
        super().__init__()
        self.convs = nn.ModuleList([ConditionedConv1d(in_channels=channels, cond_channels=cond_channels, out_channels=hidden_channels,
                                                     kernel_size=3, stride=1, padding=1),
                                    ConditionedConv1d(in_channels=hidden_channels, cond_channels=cond_channels, out_channels=hidden_channels,
                                                     kernel_size=1, stride=1, padding=0)])
        
        self.last_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.last_conv.weight.data.zero_()
        self.last_conv.bias.data.zero_()
        
    def forward(self, x, cond):
        # x: (B, C, T)
        # cond: (B, C, T)
        
        # (B, C, T)
        for i, conv in enumerate(self.convs):
            if i == 0:
                x = conv(x, cond)
            else:
                x = x + conv(x, cond)
        
        x = self.last_conv(x)
        
        return x
    
###
class Flow1d(nn.Module):
    def __init__(self, channels, cond_channels, hidden_channels):
        super().__init__()
        self.permutation = Permutation1d(channels)
        self.non_linear = NonLinear1d(channels//2, cond_channels, hidden_channels)
        
    def forward(self, x, cond):

        # Permutation
        y, log_det_W = self.permutation(x)
        # Split
        y1, y2 = split(y)
        # Transform
        m = self.non_linear(y1, cond)
        y2 = y2 + m
        
        # Merge
        y = merge(y1, y2)
        # Log-Determinant
        log_det = log_det_W
        
        return y, log_det
    
    def set_inverse(self):
        self.permutation.set_inverse()
    
    def inverse(self, y, cond):
        # Split
        x1, x2 = split(y)
        # Inverse-Transform
        m = self.non_linear(x1, cond)
        x2 = x2 - m
        # Merge
        x = merge(x1, x2)
        # Inverse-Permutation
        x = self.permutation.inverse(x)
        
        return x
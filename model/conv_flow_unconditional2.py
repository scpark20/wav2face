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
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0):
        super().__init__()
        self.c = out_channels
        self.filter_out = nn.Conv1d(in_channels=in_channels, out_channels=out_channels*2, 
                                    kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False)      
        self.filter_out.weight.data.normal_(0, 0.02)
        
    def forward(self, x):
        # x: (B, C, T)
        y = self.filter_out(x)
        y = F.tanh(y[:, :self.c]) * torch.sigmoid(y[:, self.c:])
        
        return y
    
###
class NonLinear1d(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super().__init__()
        self.in_layer = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.main_convs = nn.ModuleList([Conv1d(hidden_channels, hidden_channels, kernel_size=3,\
                                                dilation=2**(l+1), padding=2**(l+1)) for l in range(n_layers)])
        self.skip_convs = nn.ModuleList([Conv1d(hidden_channels, hidden_channels, kernel_size=1)\
                                         for l in range(n_layers)])
        self.out_layer = nn.Sequential(nn.ReLU(),
                                       nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv1d(out_channels, out_channels, kernel_size=1))
        self.out_layer[-1].weight.data.zero_()
        self.out_layer[-1].bias.data.zero_()
                                       
    def forward(self, x):
        # x: (B, C, T)
        
        x = self.in_layer(x)
        
        skips = []
        for main_conv, skip_conv in zip(self.main_convs, self.skip_convs):
            y = main_conv(x)
            skip = skip_conv(y)
            skips.append(skip)
            x = x + skip
        
        y = self.out_layer(sum(skips))
        return y
            
###
class Flow1d(nn.Module):
    def __init__(self, channels, hidden_channels, n_layers):
        super().__init__()
        self.permutation = Permutation1d(channels)
        self.non_linear = NonLinear1d(channels//2, hidden_channels, channels//2, n_layers)
        
    def forward(self, x):

        # Permutation
        y, log_det_W = self.permutation(x)
        # Split
        y1, y2 = split(y)
        # Transform
        m = self.non_linear(y1)
        y2 = y2 + m
        
        # Merge
        y = merge(y1, y2)
        # Log-Determinant
        log_det = log_det_W
        
        return y, log_det
    
    def set_inverse(self):
        self.permutation.set_inverse()
    
    def inverse(self, y):
        # Split
        x1, x2 = split(y)
        # Inverse-Transform
        m = self.non_linear(x1)
        x2 = x2 - m
        # Merge
        x = merge(x1, x2)
        # Inverse-Permutation
        x = self.permutation.inverse(x)
        
        return x
    
###
class FlowModel(nn.Module):
    def __init__(self, channels, hidden_channels, n_layers, n_flows):
        super().__init__()
        self.channels = channels
        self.flow_layers = nn.ModuleList([Flow1d(channels, hidden_channels, n_layers) for _ in range(n_flows)])
        self.inverse_init = False
        
    def forward(self, x):
        
        z = x
        log_det = 0
        for flow_layer in self.flow_layers:
            z, dlog_det = flow_layer(z)
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
    
    def inference(self, z):
        if not self.inverse_init:
            self.inverse_init = True
            self.set_inverse()

        x = self.inverse(z)
        return x
        
    def inverse(self, z):
        x = z
        for flow_layer in reversed(self.flow_layers):
            x = flow_layer.inverse(x)
        return x
    
    def set_inverse(self):
        for flow_layer in self.flow_layers:
            flow_layer.set_inverse()
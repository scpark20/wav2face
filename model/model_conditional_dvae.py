import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, z_dim, kernel_size, n_blocks, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(256, h_dim)
        self.in_layer = nn.Conv1d(in_dim, h_dim, kernel_size=1)
        self.encoder = MelEncoder(out_dim, h_dim, kernel_size, n_blocks, n_layers)
        self.decoder = MelDecoder(h_dim, h_dim, z_dim, out_dim, kernel_size, n_blocks, n_layers)
            
    def forward(self, x, y, sid):
        # x : (b, c, t)
        # y : (b, c, t)
        # sid : (b,)
        
        cond = self.in_layer(x) + self.embedding(sid).unsqueeze(-1)
        encs = self.encoder(y)
        encs = list(reversed(encs))
        y_pred, kl_divs = self.decoder(encs, cond)
        
        # Reconstruction Loss
        dim = y.size(1) * y.size(2)
        recon_loss = torch.mean(((y_pred - y) ** 2).sum(dim=[1, 2])) / dim
        
        # KL-Divergence Loss
        kl_loss = None
        for kl in kl_divs:
            kl_loss = kl.sum(dim=[1, 2]) if kl_loss is None else kl_loss + kl.sum(dim=[1, 2])
        kl_loss = torch.mean(kl_loss) / dim
                
        data = {'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'y_pred': y_pred,
               }
        return data
    
    def inference(self, x, sid):
        # x : (b, c, t)
        
        cond = self.in_layer(x) + self.embedding(sid).unsqueeze(-1)
        y_pred = self.decoder.inference(cond)
        return y_pred
    
class Conv1d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels=None,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 zero_weight=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        if zero_weight:
            self.conv.weight.data.zero_()
        else:
            self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class GatedConv1d(nn.Module):
    def __init__(self, in_channels, 
                 out_channels=None,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 zero_weight=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv1d(in_channels, out_channels*2, kernel_size, stride, padding, dilation, bias=bias)
        if zero_weight:
            self.conv.weight.data.zero_()
        else:
            self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.split(x.shape[1]//2, dim=1)
        x = torch.tanh(x1) + torch.sigmoid(x2)
        return x

class WN(nn.Module):
    def __init__(self, channels, kernel_size=5, n_blocks=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.convs.append(GatedConv1d(channels, kernel_size=kernel_size, dilation=dilation, padding=padding))
            self.residuals.append(Conv1d(channels))
            self.skips.append(Conv1d(channels))
        self.out_layer = nn.Sequential(nn.ReLU(),
                                       Conv1d(channels),
                                       nn.ReLU(),
                                       Conv1d(channels))
        
    def forward(self, x):
        # x : (b, c, t)
        skips = []
        for conv, residual, skip in zip(self.convs, self.residuals, self.skips):
            y = conv(x)
            x = x + residual(y)
            skips.append(skip(y))
        x = self.out_layer(sum(skips))
        return x

class MelEncoder(nn.Module):
    def __init__(self, in_dim, enc_dim, kernel_size=3, n_blocks=5, n_layers=8):
        super().__init__()
        
        self.in_layer = Conv1d(in_dim, enc_dim)
        self.layers = nn.ModuleList([WN(enc_dim, kernel_size, n_blocks) for _ in range(n_layers)])
        
    def forward(self, x):
        x = self.in_layer(x)
        xs = []
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        return xs    

class DecoderLayer(nn.Module):
    def __init__(self, dec_dim, enc_dim, z_dim, kernel_size=3, n_blocks=5):
        super().__init__()
        
        self.z_dim = z_dim
        self.q = Conv1d(dec_dim+enc_dim, z_dim*2, zero_weight=True)
        self.out = WN(dec_dim, kernel_size, n_blocks)
        
    def _get_kl_div(self, q_params):
        p_mean = 0
        p_logstd = 0
        q_mean = q_params[0]
        q_logstd = q_params[1]
        
        return -q_logstd + 0.5 * (q_logstd.exp() ** 2 + q_mean ** 2) - 0.5
    
    def _sample_from_q(self, q_params):
        mean = q_params[0]
        logstd = q_params[1]
        sample = mean + mean.new(mean.shape).normal_() * logstd.exp()
        
        return sample
    
    def _sample_from_p(self, tensor, shape, temperature=1.0):
        sample = tensor.new(*shape).normal_() * temperature
        return sample
        
    def forward(self, x, enc, cond):
        # x : main input
        # enc : input from encoder
        # cond : condition (text)
        
        if x is None:
            y = x = cond
        else:
            y = x + cond
        
        q_params = self.q(torch.cat([y, enc], dim=1)).split(self.z_dim, dim=1)
        kl_div = self._get_kl_div(q_params)
        z = self._sample_from_q(q_params)
        y[:, :self.z_dim] += z
        y = x + self.out(y)
        
        return y, kl_div
    
    def inference(self, x, cond, temperature=1.0):
        if x is None:
            y = x = cond
        else:
            y = x + cond

        z = self._sample_from_p(x, (x.size(0), self.z_dim, x.size(2)), temperature)
        y[:, :self.z_dim] += z
        y = x + self.out(y)
        
        return y
        

class MelDecoder(nn.Module):
    def __init__(self, dec_dim, enc_dim, z_dim, out_dim, kernel_size=3, n_blocks=5, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dec_dim, enc_dim, z_dim, kernel_size, n_blocks)\
                                     for _ in range(n_layers)])
        self.out_layer = Conv1d(dec_dim, out_dim)
        
    def forward(self, encs, cond):
        x = None
        kl_divs = []
        for layer, enc in zip(self.layers, encs):
            x, kl_div = layer(x, enc, cond)
            kl_divs.append(kl_div)
        x = self.out_layer(x)
        return x, kl_divs
    
    def inference(self, cond, temperature=1.0):
        x = None
        for layer in self.layers:
            x = layer.inference(x, cond, temperature)
        x = self.out_layer(x)
        return x
        



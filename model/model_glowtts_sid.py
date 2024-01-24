from glowtts.models import TextEncoder
from glowtts.models import FlowSpecDecoder
import glowtts.attentions as attentions
import glowtts.commons as commons
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=6, window_size=4):
        super().__init__()
        self.prenet = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.encoder = attentions.Encoder(
                       hidden_channels=hidden_dim,
                       filter_channels=hidden_dim*4,
                       n_heads=2,
                       n_layers=n_layers,
                       kernel_size=3,
                       p_dropout=0.1,
                       window_size=window_size,
                       block_length=None,
                       )
        self.postnet = nn.Conv1d(hidden_dim, out_dim*2, kernel_size=1)
        
    def forward(self, x):
        x = self.prenet(x)
        x_mask = torch.ones(x.shape[0], 1, x.shape[2]).to(x.device)
        x = self.encoder(x, x_mask)
        x = self.postnet(x)
        m, logs = x.split(x.shape[1]//2, dim=1)
        return m, logs
    
class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_split, sid_dim, dilation_rate=1, n_layers=4):
        super().__init__()
        self.decoder = FlowSpecDecoder(in_channels=in_dim, 
                       hidden_channels=hidden_dim,
                       kernel_size=5, 
                       dilation_rate=dilation_rate,
                       n_blocks=12, 
                       n_layers=n_layers,
                       p_dropout=0.05, 
                       n_split=n_split,
                       n_sqz=1,
                       sigmoid_scale=False,
                       gin_channels=sid_dim)
        
    def forward(self, x, sid):
        z_mask = torch.ones(x.shape[0], 1, x.shape[2]).to(x.device)
        z, logdet = self.decoder(x, z_mask, g=sid.unsqueeze(-1))
        return z, logdet
    
    def inference(self, z, sid):
        z_mask = torch.ones(z.shape[0], 1, z.shape[2]).to(z.device)
        x, _ = self.decoder(z, z_mask, g=sid.unsqueeze(-1), reverse=True)
        return x
    
def mle_loss(z, m, logs, logdet):
    l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet) # log jacobian determinant
    l = l / torch.sum(torch.ones_like(z)) # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
    return l

class Model(nn.Module):
    def __init__(self, in_dim, enc_hidden_dim, out_dim, dec_hidden_dim, n_speaker=256, sid_dim=256, n_split=4,
                 n_encoder_layers=6, encoder_window_size=4,
                 dilation_rate=1, n_layers=4):
        super().__init__()
        if out_dim % n_split != 0:
            padded_out_dim = math.ceil(out_dim / n_split) * n_split
        else:
            padded_out_dim = out_dim
        self.out_dim = out_dim
        self.padded_out_dim = padded_out_dim
        self.sid_embedding = nn.Embedding(n_speaker, sid_dim)
        self.encoder = Encoder(in_dim, enc_hidden_dim, padded_out_dim, n_encoder_layers, encoder_window_size)
        self.decoder = Decoder(padded_out_dim, dec_hidden_dim, n_split, sid_dim, dilation_rate, n_layers)
        self.n_split = n_split
        
    def forward(self, x, y, sid):
        sid = self.sid_embedding(sid)
        if y.shape[1] != self.padded_out_dim:           
            y = F.pad(y, (0, 0, 0, self.padded_out_dim - self.out_dim))
        m, logs = self.encoder(x)
        z, logdet = self.decoder(y, sid)
        loss = mle_loss(z, m, logs, logdet)
        data = {'m': m,
                'logs': logs,
                'z': z,
                'loss': loss,
               }
        return data
    
    def inference(self, x, sid, temperature=1.0):
        sid = self.sid_embedding(sid)
        m, logs = self.encoder(x)
        z = m + torch.randn_like(logs.exp())*temperature
        y = self.decoder.inference(z, sid)
        y = y[:, :self.out_dim]
        return y

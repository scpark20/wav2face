import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbhg import CBHG
from .bottle import Bottle

class LatentDecoder(nn.Module):
    def __init__(self, K, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(K, latent_dim)
        self.prenet = nn.Sequential(nn.Linear(latent_dim*2, latent_dim),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim, latent_dim))
        self.lstm = nn.LSTMCell(latent_dim, latent_dim)
        self.out_linear = nn.Linear(latent_dim, K)
        
    def forward(self, zi, c):
        # zi : (b, t)
        # c : (b, c, t)
        
        # (b, c, t)
        z = self.embedding(zi).transpose(1, 2)
        z = F.pad(z[:, :, :-1], (1, 0))
        states = None
        zi_preds = []
        for i in range(z.shape[2]):
            lstm_input = self.prenet(torch.cat([z[:, :, i], c[:, :, i]], dim=1))
            states = self.lstm(lstm_input, states)
            zi_pred = self.out_linear(states[0])
            zi_preds.append(zi_pred)          
        # (b, c, t)
        zi_pred = torch.stack(zi_preds, dim=2)
        return zi_pred
    
    def inference(self, c, n_beams=1, n_depth=1):
        # c : (b, c, t)
        
        # Init state
        z = torch.zeros(c.shape[0], self.latent_dim).to(c.device)
        states = None
        zis = []
        
        i = 0
        while i < c.shape[2]:
            states_list = [states for _ in range(n_beams)]
            score_list = [None for _ in range(n_beams)]
            z_list = [z for _ in range(n_beams)]
            zis_list = [None for _ in range(n_beams)]
            
            for k in range(n_beams):
                current_states = states_list[k]
                current_score = 0
                current_z = z_list[k]                
                current_zi_list = []
                current_i = i
                
                for j in range(n_depth):
                    if current_i >= c.shape[2]:
                        break
                    lstm_input = self.prenet(torch.cat([current_z, c[:, :, current_i]], dim=1))
                    current_states = self.lstm(lstm_input, current_states)
                    # (1, c)
                    logit = self.out_linear(current_states[0])
                    # (1, c)
                    prob = F.softmax(logit, dim=1)
                    # (1, 1)
                    zi = torch.multinomial(prob, 1)
                    current_zi_list.append(zi)
                    current_z = self.embedding(zi[:, 0])
                    current_score = current_score + logit[0, zi[0, 0].item()].item()
                    current_i += 1
                    
                states_list[k] = current_states
                score_list[k] = current_score
                z_list[k] = current_z
                zis_list[k] = current_zi_list
                
            # Find max
            max_score = max(score_list)
            max_index = score_list.index(max_score)
            
            # Choose state
            states = states_list[max_index]
            z = z_list[max_index]
            zis = zis + zis_list[max_index]
            
            # Increment
            i = current_i
                
        # (b, t)
        zi = torch.cat(zis, dim=1)
        return zi
            
class Model(nn.Module):
    def __init__(self, in_dim, out_dim, K, latent_dim):
        super().__init__()
        # out -> ze
        self.out_encoder = CBHG(out_dim, latent_dim)
        # ze -> zq
        self.bottle = Bottle(K, latent_dim)
        # zq -> out
        self.out_decoder = CBHG(latent_dim, out_dim)
        
        # x -> c
        self.in_encoder = CBHG(in_dim, latent_dim)
        # zq auto-regressive
        self.latent_decoder = LatentDecoder(K, latent_dim)
            
    def _quantize(self, ze):
        b, _, t = ze.size()
        ze = ze.transpose(1, 2).reshape(b*t, -1)
        bottle_outputs = self.bottle(ze)
        zq = bottle_outputs['zq'].reshape(b, t, -1).transpose(1, 2)
        bottle_outputs['zq'] = zq
        zi = bottle_outputs['zi'].reshape(b, t)
        bottle_outputs['zi'] = zi
        return bottle_outputs
    
    def _embedding(self, zi):
        b, t = zi.size()
        # (b*t)
        zi = zi.reshape(b*t)
        # (b*t, c)
        zq = torch.index_select(self.bottle.codebook, 0, zi)
        # (b, c, t)
        zq = zq.reshape(b, t, -1).transpose(1, 2)
        return zq
        
    def forward(self, x, y):
        # x : (b, c, t)
        # y : (b, c, t)
        
        '''Auto-Encoding'''
        
        # (b, c, t)
        ze = self.out_encoder(y)
        # (b, c, t)
        bottle_outputs = self._quantize(ze)
        # (b, c, t)
        y_pred = self.out_decoder(bottle_outputs['zq'])
        # (,)
        auto_encoding_loss = F.l1_loss(y_pred, y)
            
        '''Latent Decoding'''
        # (b, c, t)
        c = self.in_encoder(x)
        # (b, c, t)
        zi_pred = self.latent_decoder(bottle_outputs['zi'], c)
        zi_prediction_loss = F.cross_entropy(zi_pred, bottle_outputs['zi'])
        
        data = {'auto_encoding_loss': auto_encoding_loss,
                'commit_loss': bottle_outputs['commit_loss'],
                'zi_prediction_loss': zi_prediction_loss
               }
            
        return data
    
    def inference(self, x, n_beams=1, n_depth=1):
        # x : (b, c, t)
        
        '''Latent Decoding'''
        # (b, c, t)
        c = self.in_encoder(x)
        print('c shape :', c.shape)
        # (b, t)
        zi = self.latent_decoder.inference(c, n_beams, n_depth)
        print('zi shape :', zi.shape)
        # (b, c, t)
        zq = self._embedding(zi)
        print('zq shape :', zq.shape)
        y = self.out_decoder(zq)
        print('y shape :', y.shape)
        return y

   
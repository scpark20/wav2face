import numpy as np
import pandas as pd
import torch
import librosa
from .functional import f

class LipsDataset(torch.utils.data.Dataset):
    def __init__(self, wav_files, csv_files, n_frames, n_mels=80, sr=24000, fps=30, perturb=False):
        super().__init__()
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.sr = sr
        self.n_fft = int(sr/fps)
        self.perturb = perturb
        
        wavs = []
        targets = []
        for wav_file, csv_file in zip(wav_files, csv_files):
            df = pd.read_csv(csv_file)
            df = df.drop(columns=['Seconds'])
            target = np.array(df)
            target[np.isnan(target)] = 0
            targets.append(target)
            
            wav, _ = librosa.load(wav_file, sr=24000, res_type='polyphase')
            wav = wav[:len(df)*self.n_fft]
            if len(wav) < len(df)*self.n_fft:
                wav = np.pad(wav, (0, len(df)*self.n_fft-len(wav)))
            wavs.append(wav)
            
        self.wav = np.concatenate(wavs, axis=0)
        self.target = np.concatenate(targets, axis=0)
        
    def __getitem__(self, index):
        start = index
        end = start + self.n_frames
        wav = self.wav[start*self.n_fft:end*self.n_fft]
        if self.perturb:
            wav = f(torch.Tensor(wav), self.sr).numpy()
        wav = wav / max(abs(wav))
        mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_fft=self.n_fft, hop_length=self.n_fft, center=False, n_mels=self.n_mels)
        mel = np.log10(mel + 1e-5).T
        target = self.target[start:end]
        data = {'input': mel,
                'output': target,
                'wav': wav}
        return data
    
    def __len__(self):
        return len(self.target) - self.n_frames
        
class Collate:
    def __init__(self, n_frames, n_mels):
        self.n_frames = n_frames
        self.n_mels = n_mels
        
    def __call__(self, batch):
        
        inputs = np.zeros((len(batch), self.n_frames, self.n_mels))
        outputs = np.zeros((len(batch), self.n_frames, 61))
        wavs = np.zeros(len(batch), len(batch[0]['wav'])) 
        for i, b in enumerate(batch):
            inputs[i, :] = b['input']
            outputs[i, :] = b['output']
            wavs[i, :] = b['wav']
            
        data = {'inputs': torch.Tensor(inputs),
                'outputs': torch.Tensor(outputs),
                'wavs': torch.Tensor(wavs)
               }
        return data
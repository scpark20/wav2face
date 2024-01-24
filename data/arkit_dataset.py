import numpy as np
import pandas as pd
import torch
import librosa
from .functional import f
import torch.nn.functional as F

class LipsDataset(torch.utils.data.Dataset):
    def __init__(self, file, n_mels=80, length=400, sid=0,
                 perturb=True, mel=False, amp_aug=False, noise_aug=False, length_aug=False):
        super().__init__()
        self.sid = sid
        self.perturb = perturb
        self.amp_aug = amp_aug
        self.noise_aug = noise_aug
        self.length_aug = length_aug
        data = np.load(file, allow_pickle=True).item()
        wav = data['wav']
        self.wav = librosa.resample(wav, orig_sr=44100, target_sr=24000, res_type='polyphase')
        blend = data['blendshapes']
        
        self.file = file
        self.blend = blend
        self.n_mels = n_mels
        self.length = length
        self.mel = mel
        
    def __getitem__(self, index):
        length = self.length
        if self.length_aug:
            length = int(length * np.random.uniform(0.5, 2.0))
            if index + length > len(self.blend):
                length = self.length
        blend = self.blend[index:index+length]
        wav = self.wav[index*800:(index+length)*800]
        if length != self.length:
            # (1, c, t)
            blend = torch.Tensor(blend).T.unsqueeze(0)
            blend = F.interpolate(blend, size=self.length, mode='linear')
            # (t, c)
            blend = blend[0].T.numpy()
            # (1, 1, t)
            wav = torch.Tensor(wav).unsqueeze(0).unsqueeze(0)
            wav = F.interpolate(wav, size=self.length*800, mode='linear')
            wav = wav[0, 0].numpy()
            
        if self.perturb:
            wav = f(torch.Tensor(wav), 24000).numpy()
        wav = wav / max(abs(wav))
        if self.amp_aug:
            wav = wav * np.random.uniform(0.1, 2.0)
        if self.noise_aug:
            noise_factor = np.random.uniform(1e-4, 1e-2)
            noise = noise_factor * np.random.randn(len(wav))
            wav = wav + noise
        mel = None
        if self.mel:
            mel = librosa.feature.melspectrogram(y=wav, sr=24000, n_fft=2048, hop_length=800, win_length=2048, n_mels=self.n_mels)
            mel = np.log10(mel + 1e-5).T
            if len(mel) > len(blend):
                mel = mel[:len(blend)]
            if len(mel) < len(blend):
                mel = np.pad(mel, ((0, len(blend)-len(mel)), (0, 0)))
                
        data = {'file': self.file,
                'index': index,
                'sid': self.sid,
                'mel': mel,
                'wav': wav,
                'blend': blend}
        return data
        
    def __len__(self):
        return len(self.blend) - self.length
    
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.ratios = [1/len(datasets) for _ in range(len(datasets))]
        
    def __getitem__(self, index):
        while True:
            u = np.random.uniform()
            use_dataset = None
            ratio_sum = 0
            for dataset, ratio in zip(self.datasets, self.ratios):
                ratio_sum += ratio
                if u < ratio_sum:
                    use_dataset = dataset
                    break
            if use_dataset is None:
                use_dataset = dataset
            try:    
                data = use_dataset[index%len(use_dataset)]
            except:
                index += (index + 1)
                continue
            break
            
        return data
        
    def __len__(self):
        return max([len(dataset) for dataset in self.datasets])

class CombinedCollate:
    def __call__(self, batch):
        mel_exists = batch[0]['mel'] is not None
        if mel_exists:
            mel_lengths = [len(b['mel']) for b in batch]
        wav_lengths = [len(b['wav']) for b in batch]
        blend_lengths = [len(b['blend']) for b in batch]
        
        file = []
        index = []
        if mel_exists:
            mel = np.zeros((len(batch), max(mel_lengths), batch[0]['mel'].shape[1]))
        else:
            mel = None
        wav = np.zeros((len(batch), max(wav_lengths)))
        blend = np.zeros((len(batch), max(blend_lengths), batch[0]['blend'].shape[1]))
        sid = np.zeros(len(batch))
        
        for i, b in enumerate(batch):
            file.append(b['file'])
            index.append(b['index'])
            if mel_exists:
                mel[i, :len(b['mel'])] = b['mel']
            wav[i, :len(b['wav'])] = b['wav']
            blend[i, :len(b['blend'])] = b['blend']
            sid[i] = b['sid']

        outputs = {'file': file,
                   'index': index,
                   'mel': mel,
                   'wav': wav,
                   'blend': blend,
                   'sid': sid,
                  }

        return outputs

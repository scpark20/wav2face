from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.params import File
from starlette.responses import FileResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import librosa

# Hyperparams
n_mels = 8
n_outputs = 61
n_frames = 400
sr = 24000
fps = 30
n_fft = int(sr/fps)
device = 'cuda:0'

def load_models():
    
    from model.model_vqvae import Model
    device = 'cuda:0'

    # Model
    model = Model(in_dim=n_mels, out_dim=n_outputs, K=16, latent_dim=128)
    model = model.to(device)

    path = '/data/scpark/save/lips/train05.24-2/save_1000'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model

app = FastAPI()
model = load_models()

@app.post("/wav_to_lips")
async def wav_to_lips(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        wav_file = 'temp.wav'
        with open(wav_file, "wb") as f:
            f.write(contents)
            
        wav, _ = librosa.load(wav_file, sr=sr, res_type='polyphase')
        wav = wav / max(abs(wav))
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=n_fft, center=False, n_mels=n_mels)
        mel = np.log10(mel + 1e-5)
        mel_tensor = torch.Tensor(mel).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred = model.inference(mel_tensor, n_beams=1, n_depth=1)
            pred = pred[0].T.data.cpu().numpy()
        pred = np.pad(pred[7:], (0, 7))
        return pred.tolist()
    
    except Exception as e:
        return 0

@app.post("/predict")
async def predict(sentence: str):
    # 여기에 문장을 처리하는 코드를 작성합니다.
    return {"result": "success"}

@app.post("/echo")
async def echo(text: str):
    return {"text": text}
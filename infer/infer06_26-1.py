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
n_mels = 16
n_outputs = 61
n_frames = 400
device = 'cuda:0'

def load_models():
    
    from model.model_transformer_reg import Model

    device = 'cuda:0'

    # Model
    model = Model(in_dim=n_mels, h_dim=512, out_dim=n_outputs)
    model = model.to(device)
    path = '/data/scpark/save/lips/train06.26-1/save_30000'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model

app = FastAPI()
model = load_models()

@app.post("/wav_to_lips")
async def wav_to_lips(file: UploadFile = File(...), expansion_value=10, max_value=0.6):
    contents = await file.read()
    wav_file = 'temp.wav'
    with open(wav_file, "wb") as f:
        f.write(contents)

    wav, _ = librosa.load(wav_file, sr=24000, res_type='polyphase')
    wav = wav / max(abs(wav))
    mel = librosa.feature.melspectrogram(y=wav, sr=24000, n_fft=2048, hop_length=800, win_length=2048, n_mels=n_mels)
    mel = np.log10(mel + 1e-5)
    mel_tensor = torch.Tensor(mel).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model.inference(mel_tensor)
        pred = torch.sigmoid(pred)
        pred = torch.sigmoid((pred / torch.max(pred) - 0.5) * expansion_value) * max_value
        pred = pred[0].T.data.cpu().numpy()
    return pred.tolist()

@app.post("/predict")
async def predict(sentence: str):
    # 여기에 문장을 처리하는 코드를 작성합니다.
    return {"result": "success"}

@app.post("/echo")
async def echo(text: str):
    return {"text": text}
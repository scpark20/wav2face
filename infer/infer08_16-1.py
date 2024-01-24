from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.params import File
from starlette.responses import FileResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import librosa
import fairseq
from data.audio import mel_spectrogram
from functools import partial

get_mel = partial(mel_spectrogram, n_fft=2048, num_mels=80, sampling_rate=24000, hop_size=800, win_size=2048, fmin=0, fmax=None, center=False, return_spec=False)

# Hyperparams
n_mels = 80
n_outputs = 61
n_frames = 400
device = 'cuda:0'

def load_models():
    
    from model.model_ecapa_transformer_reg import Model

    # Model
    model = Model(in_dim=n_mels, h_dim=512, out_dim=n_outputs)
    model = model.to(device)
    path = '/data/scpark/save/lips/train08.16-1/save_30000'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()    
    return model

app = FastAPI()
model = load_models()

@app.post("/wav_to_lips")
async def wav_to_lips(file: UploadFile = File(...),
                      axis1: float = 0.0,
                      axis2: float = 0.0,
                      temperature: float = 0.1,
                     ):
    contents = await file.read()
    wav_file = 'temp.wav'
    with open(wav_file, "wb") as f:
        f.write(contents)

    wav, _ = librosa.load(wav_file, sr=24000, res_type='polyphase')
    wav = wav / max(abs(wav))
    mel = get_mel(torch.Tensor(wav).unsqueeze(0)).to(device)
    #speaker = torch.Tensor(np.array([axis1, axis2])).unsqueeze(0).to(device)
    # 0.0, 0.0
    # 1.0, 1.0
    # -2.0, 1.0 (good 눈감음)
    # -2.0, -1.0 (best... 입 크게 벌림, 오발음 잘 안됨)
    # 1.0, -1.0
    # -2.0, 0.0
    speaker = torch.Tensor(np.array([-2.0, 0.0])).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model.inference(mel, speaker)
        pred = pred[0].T.data.cpu().numpy()
    return pred.tolist()

@app.post("/predict")
async def predict(sentence: str):
    # 여기에 문장을 처리하는 코드를 작성합니다.
    return {"result": "success"}

@app.post("/echo")
async def echo(text: str):
    return {"text": text}
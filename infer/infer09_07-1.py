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
from functools import partial
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Hyperparams
n_mels = 1024
n_outputs = 61
n_frames = 400
device = 'cuda:0'

processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
wav2vec = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to(device)
print('wav2vec loaded')

def load_models():
    
    from model.model_ecapa_transformer_reg import Model

    # Model
    model = Model(in_dim=n_mels, h_dim=512, out_dim=n_outputs)
    model = model.to(device)
    path = '/data/scpark/save/lips/train09.07-1/save_140000'
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
    
    wav, _ = librosa.load(wav_file, sr=16000, res_type='polyphase')
    print(len(wav))
    
    wav = wav / max(abs(wav))
    with torch.no_grad():
        states = wav2vec(torch.Tensor(wav).unsqueeze(0).to(device),\
                         output_hidden_states=True).hidden_states[16].transpose(1, 2)
        states = F.interpolate(states, scale_factor=3/5, mode='linear').detach()
    # 입을 위아래로 적게 벌림    
    #axis1 = 5.94
    #axis2 = -11.96    
    # 입은 크게 벌리나 윗입술이 너무 올라감
    #axis1 = 4.94
    #axis2 = 9.39
    # 입을 위아래로 적게 벌림
    #axis1 = 4.94
    #axis2 = 0.0
    # Good
    #axis1 = 5.0
    #axis2 = 5.0
    speaker = torch.Tensor(np.array([axis1, axis2])).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.inference(states, speaker)
        pred = torch.clamp(pred, min=0, max=1)
        pred = pred[0].T.data.cpu().numpy()

    return pred.tolist()

@app.post("/predict")
async def predict(sentence: str):
    # 여기에 문장을 처리하는 코드를 작성합니다.
    return {"result": "success"}

@app.post("/echo")
async def echo(text: str):
    return {"text": text}
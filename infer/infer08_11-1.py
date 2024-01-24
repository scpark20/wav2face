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

# Hyperparams
n_mels = 768
n_outputs = 61
n_frames = 400
device = 'cuda:0'

def load_models():
    
    from model.model_glowtts_sid import Model

    device = 'cuda:0'

    # Model
    model = Model(in_dim=n_mels, enc_hidden_dim=256, out_dim=61, dec_hidden_dim=256)
    model = model.to(device)
    path = '/data/scpark/save/lips/train08.11-1/save_110000'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    
    ckpt_path = "/Storage/speech/pretrained/contentvec/checkpoint_best_legacy_500.pt"
    hubert, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    hubert = hubert[0]
    hubert = hubert.to(device)
    hubert.eval()

    return model, hubert

def get_hubert_feature(hubert, wav):
    with torch.no_grad():
        # (b, t, c)
        wav = torch.Tensor(wav[None, :]).to(device)
        feature = hubert.extract_features(wav, output_layer=12)[0]
        return feature.transpose(1, 2)

app = FastAPI()
model, hubert = load_models()

@app.post("/wav_to_lips")
async def wav_to_lips(file: UploadFile = File(...),
                      sid: int = 1,
                      temperature: float = 0.1,
                     ):
    contents = await file.read()
    wav_file = 'temp.wav'
    with open(wav_file, "wb") as f:
        f.write(contents)

    wav, _ = librosa.load(wav_file, sr=16000, res_type='polyphase')
    wav = wav / max(abs(wav))
    feature = get_hubert_feature(hubert, wav)
    feature = F.interpolate(feature, scale_factor=3/5, mode='linear')
    sid = torch.Tensor([sid]).long().to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model.inference(feature, sid=sid, temperature=temperature)
        pred = pred[0].T.data.cpu().numpy()
    return pred.tolist()

@app.post("/predict")
async def predict(sentence: str):
    # 여기에 문장을 처리하는 코드를 작성합니다.
    return {"result": "success"}

@app.post("/echo")
async def echo(text: str):
    return {"text": text}
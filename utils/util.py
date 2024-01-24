import torch
import librosa
import numpy as np

def save(save_dir, step, model, ema_model, optimizer):
    path = save_dir + 'save_' + str(step)
    torch.save({'step': step,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state_dict': optimizer.state_dict()}, 
                path)
    print('saved', path)
    
def load(save_dir, step, model, ema_model, optimizer, strict=True):
    path = save_dir + 'save_' + str(step)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=strict)
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass
    print('loaded', path)
    return step, model, ema_model, optimizer

def get_size(model):
    return sum([param.nelement() * param.element_size() for param in model.parameters()]) / 1024**2
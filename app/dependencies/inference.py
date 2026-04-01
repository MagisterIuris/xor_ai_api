import torch 
from app.services.inference_service import InferenceService
from ml.model.xor_net import XORNet
from fastapi import Depends
from functools import lru_cache
from config import MODEL_PATH


@lru_cache
def get_model ():
    model = XORNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model 


def get_inference_service (model = Depends(get_model)): 
    return InferenceService(model)
import torch 
from ml.model import XORNet
from app.schemas import InputData


class InferenceService:  
    def __init__(self, model: XORNet): 
        self.model = model 

    def infer(self, data: InputData): 
        inputs = torch.tensor([[data.x1, data.x2]])
        with torch.no_grad():
            output = self.model(inputs)
            prediction = int((output > 0.5).item())
        return ([data.x1, data.x2], output.item(), prediction)  

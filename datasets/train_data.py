import torch 

data = torch.tensor ([
    [0, 0], 
    [0, 1], 
    [1, 0],  
    [1, 1]  
], dtype=torch.float32)

labels = torch.tensor([
    [0],
    [1],
    [1],
    [0],
], dtype=torch.float32)
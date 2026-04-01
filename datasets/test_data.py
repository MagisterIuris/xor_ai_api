import torch 

test_data = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.99, 0.01],
    [0.01, 0.99],
    [0.5, 0.5],
    [0.9, 0.9],
    [1.2, 0.0],
    [0.0, 1.2]
], dtype=torch.float32)
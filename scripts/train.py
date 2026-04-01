import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt
from ml.model import XORNet
from config import MODEL_PATH
from datasets import data, labels

XOR_model = XORNet() 

critere = nn.BCELoss() 
optimizer = optim.SGD(XOR_model.parameters(), lr= 0.4) 

loss_history = []
for epoch in range (100000): 
    output = XOR_model(data) 
    print("Output :", output)
    loss = critere(output, labels)
    loss_history.append(loss.item())

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.grid(True)
plt.show()


torch.save(XOR_model.state_dict(), MODEL_PATH)
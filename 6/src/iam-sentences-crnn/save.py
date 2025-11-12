import os
import torch

dir = "model"

def save_model(model, filename):
    if not os.path.exists(dir):  
        os.mkdir(dir)

    torch.save(model.state_dict(), f"{dir}/{filename}")

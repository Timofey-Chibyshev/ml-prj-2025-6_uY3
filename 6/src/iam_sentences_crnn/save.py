import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.join(script_dir, "model")

def save_model(model, filename):
    if not os.path.exists(dir):  
        os.mkdir(dir)

    torch.save(model.state_dict(), os.path.join(dir, filename))

import sys
import torch
import numpy as np
from model import MnistNumbersCNN
from preprocess import preprocess_image
from save import dir

if len(sys.argv) <= 1:
    print("Usage: python predict.py <picture-path>")

pic_path = sys.argv[1]
model_name = "mnist_numbers_cnn"

model = MnistNumbersCNN()
try:
    print("Loading model...")
    model.load_state_dict(
        torch.load(f"{dir}/{model_name}.ckpt")
    )
    print("Model has been loaded successfully!\n")
except FileNotFoundError:
    print(f"No {model_name}.ckpt file exists!")
    exit(-1)

image = preprocess_image(pic_path)
image = np.array([image])
image_tensor = torch.Tensor(image)

model.eval()
with torch.no_grad():
    output = model(image_tensor)
    output = torch.nn.functional.softmax(output, 0)
    probability, prediction = torch.max(output, 0)

    print(output.data)
    print(f"Predicted {prediction} with {probability * 100 : .1f}% probability")

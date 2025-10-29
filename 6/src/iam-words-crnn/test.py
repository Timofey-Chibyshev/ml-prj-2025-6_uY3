from data_loader import loader
import numpy as np
import torch
import matplotlib.pyplot as plt
from alphabet import greedy_ctc_decoder
import sys
from preprocess_dataset import preprocess_word_png

if len(sys.argv) < 2:
    print("Usage: python test.py <filename>")
    exit(0)

filename = sys.argv[1]

x_test, y_test = loader.load_test_data()
model = loader.load_model()

img = preprocess_word_png(f"example/{filename}")

img = np.array([img])
img = np.array([img])
img = torch.Tensor(img)

output = model(img)
output = output[0]

_, indices = torch.max(output, 1)

ans = greedy_ctc_decoder(indices)

plt.imshow(img[0][0], cmap="gray", vmin=0, vmax=255)
plt.title(ans)
plt.show()

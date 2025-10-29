from data_loader import loader
import numpy as np
import torch
import matplotlib.pyplot as plt
from alphabet import greedy_ctc_decoder
import random

nrows = 6
ncols = 2
nsamples = nrows * ncols

x_test, y_test = loader.load_test_data()
num_test = len(y_test)

model = loader.load_model()

n = random.sample(range(0, num_test), nsamples)

x = x_test[n]
y = y_test[n]
x = np.array([np.array([channel]) for channel in x])

x = torch.Tensor(x)

output_raw = model(x)
output = np.empty((nsamples,), dtype='<U32')
for (idx, sample) in enumerate(output_raw):
    _, indices = torch.max(sample, 1)

    word = greedy_ctc_decoder(indices)
    output[idx] = word


plt.figure(figsize=(10, 6))
for (idx, word) in enumerate(output):
    plt.subplot(nrows, ncols, idx + 1)
    plt.imshow(x[idx][0], cmap="gray", vmin=0, vmax=255)
    plt.title(f"[{y[idx]}]: {word}")

plt.tight_layout()
plt.show()

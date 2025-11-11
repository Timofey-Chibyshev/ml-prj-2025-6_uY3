from data_loader import loader
import numpy as np
import torch
import matplotlib.pyplot as plt
from alphabet import greedy_ctc_decoder
import random

nrows = 5
ncols = 1
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
output = np.empty((nsamples,), dtype='<U128')
for (idx, sample) in enumerate(output_raw):
    _, indices = torch.max(sample, 1)

    word = greedy_ctc_decoder(indices)
    output[idx] = word


plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=1.3)
for (idx, word) in enumerate(output):
    plt.subplot(nrows, ncols, idx + 1)
    plt.imshow(x[idx][0], cmap="gray", vmin=0, vmax=255)

    title_correct = y[idx].replace("|", " ")
    title_predict = word.replace("|", " ")

    plt.title(f"[{title_correct}]: {title_predict}")

#plt.tight_layout()
plt.savefig("example/validation.png")
plt.show()

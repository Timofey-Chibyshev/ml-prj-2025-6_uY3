import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

train_loss = np.fromfile(os.path.join(script_dir, "model", "train_loss.f64"), dtype=np.float64)
train_loss = train_loss[train_loss > 0.0]

nepochs = len(train_loss)
epochs = np.arange(nepochs) + 1

val_loss = np.fromfile(os.path.join(script_dir, "model", "val_loss.f64"), dtype=np.float64)
val_loss = val_loss[val_loss > 0.0]

plt.figure()
plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)
plt.legend(labels=["train loss", "val loss"])
plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.grid()
plt.savefig(os.path.join(script_dir, "model", "loss.png"))
plt.close()

train_acc = np.fromfile(os.path.join(script_dir, "model", "train_acc.f64"), dtype=np.float64)
train_acc = train_acc[:nepochs]

val_acc = np.fromfile(os.path.join(script_dir, "model", "val_acc.f64"), dtype=np.float64)
val_acc = val_acc[:nepochs]

plt.figure()
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.legend(labels=["train accuracy", "val accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(script_dir, "model", "acc.png"))
plt.close()

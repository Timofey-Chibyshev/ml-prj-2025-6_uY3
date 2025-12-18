import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
from cerealconv import build_cerealconv

DATA_DIR = Path("data_split")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 75

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    DATA_DIR / "train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
val_data = val_gen.flow_from_directory(
    DATA_DIR / "val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_data = test_gen.flow_from_directory(
    DATA_DIR / "test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_data.class_indices)
print(f"🔹 Классов: {num_classes} → {list(train_data.class_indices.keys())}")

model = build_cerealconv(input_shape=(256, 256, 3), num_classes=num_classes)
model.summary()

checkpoint = ModelCheckpoint(
    MODELS_DIR / "best_cerealconv.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_cerealconv.png", dpi=300)
plt.close()
print("📈 Сохранён график: plots/training_cerealconv.png")

test_loss, test_acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_acc:.4f}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(train_data.class_indices.keys()),
            yticklabels=list(train_data.class_indices.keys()))
plt.title("Confusion Matrix — CerealConv")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_cerealconv.png", dpi=300)
plt.close()

print("📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_data.class_indices.keys())))

import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = Path("data")
OUT_DIR = Path("data_split")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

CLASSES = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.lower() != "etc"]

class_counts = {}
for cls in CLASSES:
    n = len(list((DATA_DIR / cls).glob("*")))
    class_counts[cls] = n

plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Количество изображений")
plt.title("Распределение изображений по классам (до разбиения)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution_before_split.png", dpi=300)
plt.close()

print("📊 Сохранён график: plots/class_distribution_before_split.png")

random.seed(42)
for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    images = list(cls_dir.glob("*"))
    random.shuffle(images)

    n = len(images)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_imgs in splits.items():
        out_dir = OUT_DIR / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy(img, out_dir / img.name)

print("✅ Датасет успешно разделён на train/val/test.")

split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

for split_name in ["train", "val", "test"]:
    for cls in CLASSES:
        path = OUT_DIR / split_name / cls
        split_counts[split_name][cls] = len(list(path.glob("*")))

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(CLASSES))
width = 0.25

ax.bar([i - width for i in x], [split_counts["train"][cls] for cls in CLASSES],
       width, label="Train", color="#6baed6")
ax.bar(x, [split_counts["val"][cls] for cls in CLASSES],
       width, label="Val", color="#9ecae1")
ax.bar([i + width for i in x], [split_counts["test"][cls] for cls in CLASSES],
       width, label="Test", color="#c6dbef")

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_ylabel("Количество изображений")
ax.set_title("Распределение изображений по классам в train/val/test")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution_after_split.png", dpi=300)
plt.close()

print("📈 Сохранён график: plots/class_distribution_after_split.png")
print("🎉 Всё готово — можно переходить к обучению модели!")

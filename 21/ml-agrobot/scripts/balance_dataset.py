import os
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torchvision import transforms

DATA_DIR = Path("data_split/train")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

MAX_SAMPLES = None
AUGS_PER_IMAGE = 2

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
])

class_counts = {}
for cls in DATA_DIR.iterdir():
    if cls.is_dir():
        class_counts[cls.name] = len(list(cls.glob("*")))

print("📊 Количество изображений по классам до балансировки:")
for k, v in class_counts.items():
    print(f"{k}: {v}")

if MAX_SAMPLES is None:
    MAX_SAMPLES = max(class_counts.values())

print(f"\n🎯 Целевое количество изображений на класс: {MAX_SAMPLES}\n")

for cls in tqdm(class_counts.keys(), desc="Балансировка классов"):
    cls_dir = DATA_DIR / cls
    images = list(cls_dir.glob("*"))
    current_count = len(images)

    if current_count >= MAX_SAMPLES:
        continue

    needed = MAX_SAMPLES - current_count
    num_aug_needed = (needed // AUGS_PER_IMAGE) + 1

    for i in range(num_aug_needed):
        img_path = random.choice(images)
        with Image.open(img_path).convert("RGB") as img:
            for j in range(AUGS_PER_IMAGE):
                aug_img = augmentation(img)
                new_name = f"{img_path.stem}_aug{i}_{j}.jpg"
                aug_img.save(cls_dir / new_name)

print("✅ Аугментация завершена. Классы сбалансированы.")

new_counts = {cls.name: len(list(cls.glob("*"))) for cls in DATA_DIR.iterdir() if cls.is_dir()}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(class_counts.keys(), class_counts.values(), color="#9ecae1")
ax[0].set_title("До балансировки")
ax[0].set_ylabel("Количество изображений")
ax[0].tick_params(axis='x', rotation=45)

ax[1].bar(new_counts.keys(), new_counts.values(), color="#74c476")
ax[1].set_title("После балансировки")
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "balance_comparison.png", dpi=300)
plt.close()

print("📈 Сохранён график: plots/balance_comparison.png")
print("🎉 Готово — теперь можно переходить к обучению модели!")

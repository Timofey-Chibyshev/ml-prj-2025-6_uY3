import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()


DATA_DIR = Path("data_split")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Используется устройство: {device}")


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_test_transforms)
test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=val_test_transforms)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

class_names = train_ds.classes
num_classes = len(class_names)
print(f"🔹 Классов: {num_classes} → {class_names}")

MODEL_LIST = [
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_small"
]


def build_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model

def train_and_evaluate(model_name):
    print(f"\n🧠 ===== Обучение модели: {model_name} =====")

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader,
            desc=f"{model_name} | Эпоха {epoch+1}/{args.epochs}"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total

        print(
            f"{model_name} | Epoch [{epoch+1}/{args.epochs}] | "
            f"Train loss: {running_loss/len(train_loader):.4f} | "
            f"Val loss: {val_loss/len(val_loader):.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                MODELS_DIR / f"best_{model_name}.pt"
            )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"training_{model_name}.png", dpi=300)
    plt.close()

    model.load_state_dict(
        torch.load(MODELS_DIR / f"best_{model_name}.pt")
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    print(f"\n📋 Classification Report ({model_name}):\n")
    print(report)

    with open(
        MODELS_DIR / f"classification_report_{model_name}.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.title(f"Confusion Matrix ({model_name})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_{model_name}.png", dpi=300)
    plt.close()

    print(f"✅ {model_name} — обучение и оценка завершены")

if __name__ == "__main__":
    for model_name in MODEL_LIST:
        train_and_evaluate(model_name)

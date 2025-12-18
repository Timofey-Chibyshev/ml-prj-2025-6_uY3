import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def build_model(model_name, num_classes):
    """Строит модель с указанной архитектурой"""
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    return model

def load_test_data(data_dir, batch_size=16):
    """Загружает тестовые данные"""
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = datasets.ImageFolder(data_dir / "test", transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_ds.classes, test_ds.class_to_idx

def test_model(model_path, model_name, num_classes, data_dir, output_dir="test_results"):
    """Тестирует модель и возвращает метрики"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Используется устройство: {device}")
    
    print("📁 Загрузка данных...")
    test_loader, class_names, class_to_idx = load_test_data(Path(data_dir))
    print(f"🔹 Классы: {class_names}")
    print(f"🔹 Соответствие классов: {class_to_idx}")
    
    if num_classes != len(class_names):
        print(f"⚠️ Внимание: указано {num_classes} классов, но найдено {len(class_names)} классов в данных")
        print(f"🔹 Используется число классов из данных: {len(class_names)}")
        num_classes = len(class_names)
    
    print(f"🧠 Создание модели: {model_name}")
    model = build_model(model_name, num_classes).to(device)
    
    print(f"📥 Загрузка весов из: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Загружен checkpoint с дополнительной информацией")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Загружен state_dict модели")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None
    
    print("🧪 Начало тестирования...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Тестирование"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print("\n📊 Вычисление метрик...")
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print("📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    print(f"Модель: {model_name}")
    print(f"Файл модели: {model_path}")
    print(f"Количество классов: {num_classes}")
    print(f"Количество тестовых образцов: {len(all_labels)}")
    print("\n📊 Основные метрики:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    print("\n📋 Детальный отчет по классам:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, cmap="Blues")
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], 
                    ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=10)
    
    plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Предсказанные метки")
    plt.ylabel("Истинные метки")
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")
    plt.tight_layout()
    
    cm_path = output_path / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix сохранена: {cm_path}")
    
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy по классам")
    plt.ylim(0, 1)
    
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    class_acc_path = output_path / f"class_accuracy_{model_name}.png"
    plt.savefig(class_acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График accuracy по классам сохранен: {class_acc_path}")
    
    metrics_file = output_path / f"metrics_{model_name}.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Модель: {model_name}\n")
        f.write(f"Файл модели: {model_path}\n")
        f.write(f"Классы: {class_names}\n")
        f.write(f"Количество тестовых образцов: {len(all_labels)}\n\n")
        f.write("Метрики:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(f"💾 Полные метрики сохранены: {metrics_file}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'class_names': class_names
    }

def main():
    parser = argparse.ArgumentParser(description="Тестирование обученной модели")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к файлу с обученной моделью")
    parser.add_argument("--model_name", type=str, required=True, 
                       choices=["resnet50", "efficientnet_b0", "mobilenet_v3_small"],
                       help="Название архитектуры модели")
    parser.add_argument("--num_classes", type=int, required=True, help="Количество классов")
    parser.add_argument("--data_dir", type=str, default="data_split", help="Путь к директории с данными")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Директория для сохранения результатов")
    
    args = parser.parse_args()
    
    print("🧪 ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*50)
    
    results = test_model(
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if results is not None:
        print("\n🎉 Тестирование завершено успешно!")
    else:
        print("\n❌ Тестирование завершено с ошибками!")

if __name__ == "__main__":
    main()
    
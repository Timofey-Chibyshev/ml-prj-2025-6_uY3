import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import os

CLASS_NAMES_EN = [
    "BrownRust",
    "Healthy",
    "LeafBlight",
    "Mildew",
    "Septoria",
    "WheatBlast",
    "YellowRust"
]

CLASS_NAMES_RU = {
    "BrownRust": "Бурая ржавчина",
    "Healthy": "Без признаков болезни",
    "LeafBlight": "Пятнистость листьев",
    "Mildew": "Мучнистая роса",
    "Septoria": "Септориоз",
    "WheatBlast": "Пшеничный ожог (Blast)",
    "YellowRust": "Жёлтая ржавчина"
}

DISEASE_INFO = {
    "BrownRust": "Бурая ржавчина — грибковое заболевание, вызывающее появление коричневых пустул на листьях. Снижает фотосинтез и урожайность.",
    "Healthy": "Признаков заболеваний не обнаружено. Колос выглядит здоровым 👍",
    "LeafBlight": "Поражение бурой пятнистостью. Проявляется продолговатыми некротическими пятнами и может снижать урожай.",
    "Mildew": "Мучнистая роса — белый мучнистый налёт на поверхности листьев и колоса, вызывающий ослабление растения.",
    "Septoria": "Септориоз — грибковая болезнь, проявляется овальными пятнами с черными точками пикнид.",
    "WheatBlast": "Пшеничный ожог — опасное заболевание, вызывающее обесцвечивание и усыхание колоса. Может приводить к потере урожая.",
    "YellowRust": "Жёлтая ржавчина — ярко-жёлтые полосы пустул на листьях и колосе. Быстро распространяется в прохладную влажную погоду."
}


NUM_CLASSES = len(CLASS_NAMES_EN)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_cached_model = None
_cached_name = None


def build_model(model_name: str):
    """Создаёт архитектуру, как при обучении"""
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, NUM_CLASSES)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, NUM_CLASSES)

    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return model


async def load_model(model_name: str, weights_path: str):
    """Загружает модель и кэширует её"""
    global _cached_model, _cached_name

    if _cached_model and _cached_name == model_name:
        return _cached_model

    model = build_model(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _cached_model = model
    _cached_name = model_name

    print(f"ML: модель {model_name} загружена")

    return model


async def predict_image(model, image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred_id = torch.argmax(logits, dim=1).item()

    class_eng = CLASS_NAMES_EN[pred_id]
    name = CLASS_NAMES_RU[class_eng]
    info = DISEASE_INFO[class_eng]

    return f"{name}\n\n{info}"

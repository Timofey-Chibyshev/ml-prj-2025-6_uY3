from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from adaptive_emotion_model import AdaptiveEmotionModel
import os

# Инициализируем приложение
app = FastAPI(title="Emotion Detection API", version="1.0")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика (если папка есть)
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Загружаем модель при запуске
MODEL_DIR = "./models/ruberta_emotion_base"
model = AdaptiveEmotionModel(MODEL_DIR)


# ===== PYDANTIC MODELS =====

class PredictRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    predicted_emotion: str   # "happiness", "sadness" и т.п.
    corrected_emotion: str   # исправленная эмоция, тоже код


class FineTuneRequest(BaseModel):
    num_epochs: int = 3
    learning_rate: float = 5e-5
    use_lora: bool = True


# ===== API ENDPOINTS =====

@app.get("/")
def read_root():
    return {"message": "Emotion Detection API v1.0", "status": "running"}


@app.post("/predict")
def predict(request: PredictRequest):
    """Предсказывает эмоцию для текста."""
    try:
        result = model.predict(request.text, return_probs=True)
        # result: {"emotion": label, "confidence": float, "id": int, "probs": {...}}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def add_feedback(request: FeedbackRequest):
    """
    Добавляет feedback (исправление эмоции от пользователя).

    text: исходный текст
    predicted_emotion: строковый ярлык модели (например, "happiness")
    corrected_emotion: исправленный ярлык (например, "sadness")
    """
    try:
        success = model.add_feedback(
            request.text,
            request.predicted_emotion,
            request.corrected_emotion,
        )

        if not success:
            # неправильная метка – это проблема данных, а не 500
            raise HTTPException(
                status_code=400,
                detail=f"Invalid emotion label: {request.corrected_emotion}",
            )

        return {"success": True, "message": "Feedback added successfully"}
    except HTTPException:
        # пробрасываем 400 как есть
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback-stats")
def get_feedback_stats():
    """Возвращает статистику по feedback'ам."""
    try:
        return model.get_feedback_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotions")
def get_emotions():
    """Возвращает список всех доступных эмоций."""
    return {
        "emotions": model.config["labels"],
        "count": len(model.config["labels"]),
    }


@app.post("/fine-tune")
def fine_tune(request: FineTuneRequest):
    """Запускает дообучение модели на feedback'ах."""
    try:
        success = model.fine_tune(
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            use_lora=request.use_lora,
        )

        if success:
            return {"success": True, "message": "Fine-tuning completed successfully"}
        else:
            return {"success": False, "message": "Not enough feedback examples"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


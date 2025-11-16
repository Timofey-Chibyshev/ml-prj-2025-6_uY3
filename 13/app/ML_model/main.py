from fastapi import FastAPI, HTTPException
from model_train import train_pinn_model
from pydantic import BaseModel

app = FastAPI(title="ML Service")

# Глобальная переменная для хранения статуса выполнения
ml_status = {
    "is_running": False,
    "last_result": None,
    "error": None
}


class MLParams(BaseModel):
    num_layers: int = 4
    num_perceptrons: int = 50
    num_epoch: int = 10000
    optimizer: str = "Adam"
    loss_weights_config: str = ""  # Добавлен параметр конфигурации весов ошибок

@app.post("/run_ml_model")
async def run_ml_model(params: MLParams):
    global ml_status

    if ml_status["is_running"]:
        raise HTTPException(status_code=400, detail="ML модель уже выполняется")

    ml_status["is_running"] = True
    ml_status["error"] = None

    try:
        print(f"Запуск ML модели с параметрами: {params.num_layers} слоев, {params.num_perceptrons} нейронов, {params.num_epoch} эпох, оптимизатор: {params.optimizer}")
        print(f"Конфигурация весов ошибок: {params.loss_weights_config}")

        # Запускаем обучение модели с переданными параметрами
        results = train_pinn_model(
            num_layers=params.num_layers,
            num_perceptrons=params.num_perceptrons,
            num_epoch=params.num_epoch,
            optimizer=params.optimizer,
            loss_weights_config=params.loss_weights_config  # Передаем конфигурацию весов
        )

        if results["status"] == "error":
            ml_status["error"] = results["message"]
            ml_status["is_running"] = False
            raise HTTPException(status_code=500, detail=results["message"])
        else:
            ml_status["last_result"] = results
            ml_status["is_running"] = False
            return results

    except Exception as e:
        error_msg = f"Неожиданная ошибка: {str(e)}"
        ml_status["error"] = error_msg
        ml_status["is_running"] = False
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/ml_status")
async def get_ml_status():
    return ml_status


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ML",
        "ml_running": ml_status["is_running"]
    }
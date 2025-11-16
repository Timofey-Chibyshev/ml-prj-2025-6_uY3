from Logger import Logger
from DataLoadFromDB import PINNDataLoader
from Client_ClickHouse import client
from PINN import PINN
from PINN import init_model_params
import traceback


def train_pinn_model(num_layers, num_perceptrons, num_epoch, optimizer, loss_weights_config=""):
    print(f"Полученная конфигурация весов: '{loss_weights_config}'")
    try:
        # Инициализация загрузчика данных
        data_loader = PINNDataLoader(client)

        # Загрузка данных из ClickHouse
        X_u_train, u_train = data_loader.load_training_data()
        X_f_train = data_loader.load_collocation_points()

        if X_u_train is None or u_train is None or X_f_train is None:
            error_msg = "Ошибка при загрузке данных из ClickHouse"
            return {"status": "error", "message": error_msg}

        # Вычисление границ области
        lb, ub = data_loader.get_domain_bounds(X_u_train, X_f_train)

        print(f"Границы области: lb={lb}, ub={ub}")
        print(f"Размерности: X_u_train {X_u_train.shape}, u_train {u_train.shape}, X_f_train {X_f_train.shape}")

        # Параметры обучения с учетом выбранного оптимизатора
        lr_schedule, tf_optimizer, layers, tf_epochs = init_model_params(
            num_layers, num_perceptrons, num_epoch, optimizer
        )
        logger = Logger(frequency=200)

        # Создание и обучение модели с передачей конфигурации весов
        pinn = PINN(layers, tf_optimizer, logger, X_f_train, lb, ub)
        training_results = pinn.fit(X_u_train, u_train, tf_epochs, loss_weights_config)

        # Получаем график и статистику обучения
        training_plot = logger.get_training_plot()

        # Упрощенные результаты - только основные параметры
        results = {
            "status": "success",
            "message": "ML модель успешно обучена",
            "training_epochs": tf_epochs,
            "model_layers": len(layers),
            "num_perceptrons": num_perceptrons,
            "optimizer": optimizer,
            "best_loss": float(training_results["best_loss"]),
            "training_plot": training_plot,
            "loss_weights_used": loss_weights_config if loss_weights_config else "равные веса [1.0, 1.0]"
        }

        return results

    except Exception as e:
        error_msg = f"Ошибка при выполнении ML модели: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "error", "message": error_msg}
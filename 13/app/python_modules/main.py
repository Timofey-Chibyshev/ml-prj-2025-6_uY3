import numpy as np
from data_loader import load_data, get_values
from visualization import plot_comparison
from logger import Logger
from pinn_model import PINN
from config import layers, tf_optimizer, N_u, N_r, tf_epochs
import os

def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "sources", "Buckley_Swc_0_Sor_0_M_2.mat")
    
    if not os.path.exists(data_path):
        print(f"Ошибка: Файл не найден: {data_path}")
        print("Убедитесь, что файл данных находится в папке 'sources'")
        return
    
    print(f"Загружаем данные из: {data_path}")
    
    # Загрузка данных
    data = load_data(data_path)
    
    
    # Загрузка данных
    
    #data = load_data("/sources/Buckley_Swc_0_Sor_0_M_2.mat")
    
    # Подготовка данных
    x, t, X, T, Exact_u, X_star, u_star, X_u_train, u_train, X_f, ub, lb = get_values(data, N_u, N_r)
    
    # Инициализация логгера и модели
    logger = Logger(frequency=200)
    pinn = PINN(layers, tf_optimizer, logger, X_f, ub, lb)
    
    # Функция ошибки
    def error():
        u_pred = pinn.predict(X_star)
        return np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)
    
    logger.set_error_fn(error)
    
    # Обучение
    pinn.fit(X_u_train, u_train, tf_epochs)
    
    # Предсказание
    u_pred = pinn.predict(X_star)
    
    # Визуализация результатов
    plot_comparison(X_star, u_star, u_pred.numpy())

if __name__ == "__main__":
    main()
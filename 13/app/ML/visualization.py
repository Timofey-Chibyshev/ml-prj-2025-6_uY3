import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from typing import Dict, List, Optional
import os


class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.total_losses = []
        self.data_losses = []
        self.pde_losses = []

    def add_epoch_data(self, epoch: int, total_loss: float, data_loss: float, pde_loss: float):
        """Добавление данных эпохи для построения графика"""
        self.epochs.append(epoch)
        self.total_losses.append(total_loss)
        self.data_losses.append(data_loss)
        self.pde_losses.append(pde_loss)

    def create_training_plot(self) -> str:
        """Создание графика обучения в base64 формате"""
        if not self.epochs:
            return None

        plt.figure(figsize=(12, 8))

        # Полупрозрачная область для общих потерь
        plt.fill_between(self.epochs, self.total_losses, alpha=0.3, color='deepskyblue', label='Общие потери')
        plt.plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Общие потери')

        # Потери данных
        plt.plot(self.epochs, self.data_losses, 'r--', linewidth=1.5, label='Потери данных (MSE)')

        # Потери PDE
        plt.plot(self.epochs, self.pde_losses, 'g--', linewidth=1.5, label='Потери уравнения (PDE)')

        plt.yscale('log')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери (log scale)')
        plt.title('График обучения PINN модели')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Сохраняем в base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def get_training_stats(self) -> Dict:
        """Получение статистики обучения"""
        if not self.epochs:
            return {}

        return {
            "final_epoch": self.epochs[-1],
            "final_total_loss": float(self.total_losses[-1]),
            "final_data_loss": float(self.data_losses[-1]),
            "final_pde_loss": float(self.pde_losses[-1]),
            "min_total_loss": float(min(self.total_losses)),
            "min_data_loss": float(min(self.data_losses)),
            "min_pde_loss": float(min(self.pde_losses))
        }


class PredictionVisualizer:
    """Класс для визуализации предсказаний модели"""

    def create_prediction_plot(
            self,
            x_points: np.ndarray,
            predictions: np.ndarray,
            prediction_time: float,
            num_x_points: int,
            boundary_x: Optional[np.ndarray] = None,
            boundary_true: Optional[np.ndarray] = None,
            boundary_pred: Optional[np.ndarray] = None,
            figsize: tuple = (12, 8)
    ) -> str:
        """
        Создание графика предсказания с заливкой под кривой

        Args:
            x_points: Массив x координат
            predictions: Массив предсказанных значений
            prediction_time: Время предсказания
            num_x_points: Количество точек по X
            boundary_x: x координаты граничных точек (опционально)
            boundary_true: Истинные значения граничных точек (опционально)
            boundary_pred: Предсказанные значения граничных точек (опционально)
            figsize: Размер графика

        Returns:
            Base64 строка с изображением графика
        """
        try:
            plt.figure(figsize=figsize)

            # Заливка под кривой предсказания
            plt.fill_between(
                x_points.flatten(),
                predictions.flatten(),
                alpha=0.3,
                color='deepskyblue',
                label='Область предсказания'
            )

            # Основное предсказание
            plt.plot(
                x_points,
                predictions,
                'b-',
                linewidth=2,
                label='Предсказание PINN'
            )

            # Граничные точки, если переданы
            if (boundary_x is not None and
                    boundary_true is not None and
                    boundary_pred is not None):

                # Фактические граничные точки
                plt.scatter(
                    boundary_x,
                    boundary_true,
                    color='red',
                    s=80,
                    alpha=0.8,
                    label='Граничные точки (данные)',
                    zorder=5
                )

                # Предсказания для граничных точек
                plt.scatter(
                    boundary_x,
                    boundary_pred,
                    color='green',
                    s=80,
                    alpha=0.8,
                    label='Предсказания для граничных точек',
                    zorder=5,
                    marker='s'
                )

                # Линии разницы между фактическими и предсказанными значениями
                for i, (x_val, true_val, pred_val) in enumerate(zip(
                        boundary_x, boundary_true, boundary_pred
                )):
                    plt.plot(
                        [x_val, x_val],
                        [true_val, pred_val],
                        'r--',
                        linewidth=2,
                        alpha=0.7,
                        label='Ошибка' if i == 0 else ""
                    )

            # Устанавливаем заголовок и подписи осей
            plt.title(f'Предсказание модели в момент времени t = {prediction_time} ({num_x_points} точек)',
                     fontsize=14, fontweight='bold')
            plt.xlabel('x', fontsize=12)
            plt.ylabel(f'S(x, t={prediction_time})', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)

            # Сохраняем в base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{plot_data}"

        except Exception as e:
            print(f"Ошибка при создании графика предсказания: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_comparison_plot(
            self,
            x_points: np.ndarray,
            predictions: np.ndarray,
            true_values: np.ndarray,
            prediction_time: float,
            num_x_points: int,
            figsize: tuple = (12, 8)
    ) -> str:
        """
        Создание графика сравнения предсказаний с истинными значениями

        Args:
            x_points: Массив x координат
            predictions: Массив предсказанных значений
            true_values: Массив истинных значений
            prediction_time: Время предсказания
            num_x_points: Количество точек по X
            figsize: Размер графика

        Returns:
            Base64 строка с изображением графика
        """
        try:
            plt.figure(figsize=figsize)

            # Истинные значения
            plt.plot(
                x_points,
                true_values,
                'g-',
                linewidth=2,
                label='Истинные значения'
            )

            # Предсказания
            plt.plot(
                x_points,
                predictions,
                'b--',
                linewidth=2,
                label='Предсказания PINN'
            )

            # Область разницы
            plt.fill_between(
                x_points.flatten(),
                true_values.flatten(),
                predictions.flatten(),
                alpha=0.3,
                color='red',
                label='Ошибка предсказания'
            )

            plt.xlabel('x', fontsize=12)
            plt.ylabel(f'S(x, t={prediction_time})', fontsize=12)
            plt.title(f'Сравнение предсказаний с истинными значениями (t={prediction_time}, {num_x_points} точек)')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)

            # Сохраняем в base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{plot_data}"

        except Exception as e:
            print(f"Ошибка при создании графика сравнения: {e}")
            return None
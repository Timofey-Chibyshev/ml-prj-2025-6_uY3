import numpy as np
import pandas as pd
from pyDOE import lhs
from . import queries


class DataProcessor:

    @staticmethod
    def normalize_data(data, columns):
        normalized_data = data.copy()
        for col in columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val - min_val == 0:
                normalized_data[col] = 0
            else:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        return normalized_data


class ClickHouseManager:

    def __init__(self, client, database='db'):
        self.client = client
        self.database = database

    def clear_table(self, table_name):
        """Очищает указанную таблицу"""
        try:
            truncate_query = queries.truncate_table(self.database, table_name)
            self.client.execute(truncate_query)
            print(f"Таблица {table_name} успешно очищена")
            return True
        except Exception as e:
            print(f"Ошибка при очистке таблицы {table_name}: {e}")
            return False

    def save_to_clickhouse(self, data, table_name, data_type, clear_existing=True):
        """Сохраняет данные в ClickHouse с опцией очистки существующих данных"""
        try:
            print("Успешное подключение к ClickHouse")

            # Создаем БД и таблицу
            self.client.execute(queries.create_db(self.database))
            print(f"База данных {self.database} создана или уже существует")

            create_table_query = queries.create_table(self.database, table_name, data_type)
            self.client.execute(create_table_query)
            print(f"Таблица {table_name} создана или уже существует")

            # Очищаем таблицу если требуется
            if clear_existing:
                self.clear_table(table_name)

            # Подготавливаем данные для вставки
            if data_type == "train_points":
                data_to_insert = [
                    (float(row['x']), float(row['t']), float(row['value']))
                    for _, row in data.iterrows()
                ]
            elif data_type == "collocate_points":
                data_to_insert = [(float(row[0]), float(row[1])) for row in data]

            # Выполняем вставку
            insert_query = queries.insert_query(self.database, table_name, data_type)
            self.client.execute(insert_query, data_to_insert)

            print(f"Успешно сохранено {len(data_to_insert)} записей в ClickHouse")
            return True

        except Exception as e:
            print(f"Ошибка при работе с ClickHouse: {e}")
            return False


class DataGenerator:

    @staticmethod
    def generate_collocation_points(data, num_points=300):
        try:
            x = data['x']
            t = data['t']

            # Создаем сетку
            X, T = np.meshgrid(x, t)
            X_setka = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

            # Генерируем точки коллокации
            lb = X_setka.min(axis=0)
            ub = X_setka.max(axis=0)
            X_collocate = lb + (ub - lb) * lhs(2, num_points)

            return X_collocate

        except Exception as e:
            print(f"Ошибка при генерации данных: {e}")
            return None


class DataManager:

    def __init__(self, clickhouse_client, database='db'):
        self.processor = DataProcessor()
        self.generator = DataGenerator()
        self.db_manager = ClickHouseManager(clickhouse_client, database)

    def process_csv_data(self, filepath, num_collocation_points=300):
        try:
            # Читаем и обрабатываем CSV
            df = pd.read_csv(filepath)
            df = self.processor.normalize_data(df, ["x", "t"])

            # Сохраняем тренировочные данные (очищаем существующие)
            if not self.db_manager.save_to_clickhouse(df, "TableTrainData", "train_points", clear_existing=True):
                return False

            # Генерируем и сохраняем точки коллокации (очищаем существующие)
            collocation_data = self.generator.generate_collocation_points(df, num_collocation_points)
            if collocation_data is not None:
                return self.db_manager.save_to_clickhouse(
                    collocation_data,
                    "CollocatePointsTable",
                    "collocate_points",
                    clear_existing=True
                )

            return False

        except Exception as e:
            print(f"Ошибка при обработке CSV файла: {e}")
            return False

    def update_collocation_points(self, filepath, num_collocation_points=300):
        """Обновляет только точки коллокации без изменения тренировочных данных"""
        try:
            # Читаем существующие данные для получения границ
            df = pd.read_csv(filepath)
            df = self.processor.normalize_data(df, ["x", "t"])

            # Генерируем новые точки коллокации
            collocation_data = self.generator.generate_collocation_points(df, num_collocation_points)
            if collocation_data is not None:
                return self.db_manager.save_to_clickhouse(
                    collocation_data,
                    "CollocatePointsTable",
                    "collocate_points",
                    clear_existing=True
                )

            return False

        except Exception as e:
            print(f"Ошибка при обновлении точек коллокации: {e}")
            return False
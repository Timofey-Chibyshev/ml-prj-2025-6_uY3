import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("datasets/by_country/Angola.csv", index_col=0, parse_dates=True)

# Создание лагов для курса валюты (например, 3 предыдущих дня)
for lag in [1, 2, 3]:
    df[f'Rate_lag_{lag}'] = df['Rate'].shift(lag)

# Удаление строк с NaN (из-за лагов)
df = df.dropna()

# Разделение на признаки и целевую переменную
X = df[['Gdp', 'Import', 'Export', 'Rate_lag_1', 'Rate_lag_2', 'Rate_lag_3']].values
y = df['Rate'].values

# Нормализация
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False, random_state=42)

# Создание модели с учётом временных признаков
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучение
history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, verbose=1)

# Предсказание
y_pred = model.predict(X_test)

# Расчёт MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Визуализация (опционально)
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

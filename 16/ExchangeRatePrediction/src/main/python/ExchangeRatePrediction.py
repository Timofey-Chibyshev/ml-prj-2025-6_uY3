import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras import models
import matplotlib.pyplot as plt

from config import settings
from dataet_prepare import dataset_prepare
from util import print_separate_message

def load_and_split_df():
    # Загрузка данных
    df = pd.read_csv(settings.filepath(), index_col=0, parse_dates=True)

    # Создание лагов для курса валюты
    for lag in range(1, 4):
        df[f'Rate_lag_{lag}'] = df['Rate'].shift(lag)

    # Удаление строк с NaN (из-за лагов)
    df = df.dropna()

    # Разделение на признаки и целевую переменную
    X = df[['Gdp', 'Import', 'Export', 'Rate_lag_1', 'Rate_lag_2', 'Rate_lag_3']]
    y = df['Rate'].values

    # Нормализация
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X.loc[:, ['Gdp', 'Import', 'Export', 'Rate_lag_1', 'Rate_lag_2', 'Rate_lag_3']] = X_scaled

    # Разделение на обучающую и тестовую выборки
    return train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)


def train_and_save(X_train, y_train):
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
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Сохранение
    if not os.path.isdir(settings.save_model_dir):
        print(f"Saved models directory {settings.save_model_dir} not found")
        print(f"Creating directory {settings.save_model_dir}")
        os.mkdir(settings.save_model_dir)

    if not os.path.isfile(settings.saved_model_filepath()):
        with open(settings.saved_model_filepath(), 'x') as file:
            pass

    model.save(settings.saved_model_filepath())

    return model

def predict_and_compare(model, X_test, y_test):
    y_pred = model.predict(X_test.values)

    # Расчёт MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index.to_numpy(), y_test, label='Real')
    plt.plot(X_test.index.to_numpy(), y_pred, label='Predicted')
    plt.xlabel("Date")
    plt.ylabel("Exchange rate")
    plt.title(f"{settings.train_country} exchange rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def do_all(country: str):
    settings.train_country = country
    X_train, X_test, y_train, y_test = load_and_split_df()
    if not os.path.isfile(settings.saved_model_filepath()):
        print_separate_message("Training model")
        model = train_and_save(X_train.values, y_train)
        print_separate_message("Save model")
        print(f"Model saved to {settings.saved_model_filepath()}")
    else:
        print_separate_message("Load Model")
        model = models.load_model(settings.saved_model_filepath())
        print(f"Model loaded from file {settings.saved_model_filepath()}")

    print_separate_message("Prediction")
    predict_and_compare(model, X_test, y_test)

def menu():
    option = ""
    while not option == "0":
        print_separate_message()
        print_separate_message("MENU")
        print("Options:\n"
              + "0. Exit\n"
              + "1. Select country and predict\n"
              + "2. See optional countries list\n")
        option = input()
        if option == "0":
            continue
        if option == "1":
            print_separate_message("Select country and predict")
            country = input("Country: ")
            while not os.path.isfile(settings.filepath(country)):
                print(f"Sorry, I don't have {country} dataset")
                country = input("Country: ")
            do_all(country)
        elif option == "2":
            print_separate_message("Optional countries list")
            for filename_with_ext in os.listdir(settings.prepared_data_dir):
                full_path = os.path.join(settings.prepared_data_dir, filename_with_ext)
                if os.path.isfile(full_path):
                    name, extension = os.path.splitext(filename_with_ext)
                    print(name)
            input()
        else:
            print_separate_message("Wrong option", "!")

def main():
    print_separate_message("START")
    if not os.path.isdir(settings.prepared_data_dir):
        print(f"Datasets directory {settings.prepared_data_dir} not found")
        print(f"Creating directory {settings.prepared_data_dir}")
        os.mkdir(settings.prepared_data_dir)

    if len(os.listdir(settings.prepared_data_dir)) == 0:
        print(f"Datasets directory {settings.prepared_data_dir} is empty")
        print_separate_message("Prepare Datasets")
        dataset_prepare()
    print(f"Datasets prepared and saved to {settings.prepared_data_dir}")

    if settings.enable_console_mode:
        menu()
    else:
        do_all()
    print_separate_message("END")

    print("Goodbye")


if __name__ == "__main__":
    main()
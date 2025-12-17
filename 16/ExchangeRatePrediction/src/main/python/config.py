enable_console_mode = True

dataset_dir = "datasets/"
currency_file_name = "exchange_rates_2021_2025.csv"
import_export_file_name = "export_import_2003_2024.csv"
gdp_file_name = "gdp_2020_2025.csv"

prepared_data_dir = "by_country/"
train_country = "Russia"
file_format = ".csv"
save_model_dir = "model/"
model_file_format = ".keras"

exchange_rates_year_from = 2021
exchange_rates_year_to = 2025

gdp_year_from = 2020
gdp_year_to = 2025

import_export_year_from = 2003
import_export_year_to = 2024

# ================================= Automatic settings aggregator ===================================
class Settings:
    def __init__(self):
        self.enable_console_mode = enable_console_mode

        self.dataset_dir = dataset_dir
        self.currency_file_name = currency_file_name
        self.import_export_file_name = import_export_file_name
        self.gdp_file_name = gdp_file_name

        self.prepared_data_dir = prepared_data_dir
        self.train_country = train_country
        self.file_format = file_format
        self.save_model_dir = save_model_dir
        self.model_file_format = model_file_format

        self.train_country = train_country
        self.currency_filepath = dataset_dir + currency_file_name
        self.import_export_filepath = dataset_dir + import_export_file_name
        self.gdp_filepath = dataset_dir + gdp_file_name

        self.exchange_rates_year_from = exchange_rates_year_from
        self.exchange_rates_year_to = exchange_rates_year_to

        self.gdp_year_from = gdp_year_from
        self.gdp_year_to = gdp_year_to

        self.import_export_year_from = import_export_year_from
        self.import_export_year_to = import_export_year_to

    def filepath(self, country = ""):
        if country == "":
            country = self.train_country
        return prepared_data_dir + country + file_format
    def saved_model_filepath(self):
        return save_model_dir + self.train_country + model_file_format


settings = Settings()

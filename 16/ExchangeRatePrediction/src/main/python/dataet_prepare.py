import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pandas import DataFrame
from numpy import arange, array, nan


def get_country_exchange_rates(df: DataFrame, target_size):
    data = df.groupby('date').max()['value']
    return [*data, *([nan] * (target_size - data.shape[0]))]


def generate_linear_data_for_period(year_from: int, year_to: int, data_from: float, data_to: float) -> list[float]:
    total_days = (pd.to_datetime(f'{year_to}-01-01')
                  - pd.to_datetime(f'{year_from}-01-01')).days
    step = 1.0 * (data_to - data_from) / total_days
    if step == 0:
        return [data_to] * total_days
    result = list(arange(data_from, data_to, step))
    if len(result) < total_days:
        result = [*result, *([data_to] * (total_days - len(result)))]
    elif len(result) > total_days:
        result = result[:(total_days - len(result))]
    return result


def generate_linear_data(years: list[int], data: list[float]) -> array:
    return array(sum([generate_linear_data_for_period(years[i], years[i + 1], data[i], data[i + 1]) for i in range(len(years) - 1)], []))


def convert_to_float_if_possible(s: str):
    try:
        return float(s)
    except ValueError:
        return nan


def generate_data(df, target_size):
    generated = generate_linear_data(list(map(int, df.index.tolist())), df.tolist())
    generated = [*generated, *([nan] * (target_size - generated.shape[0]))]
    return generated


def get_export_import() -> DataFrame:
    export_import = pd.read_csv('datasets/export_import_2003_2024.csv')
    export_import = export_import[["Country", "Series Name", "2020", "2021", "2022", "2023", "2024"]]
    export_import = export_import[export_import["Series Name"].isin(['Imports of goods and services (current US$)', 'Exports of goods and services (BoP, current US$)'])]
    export_import["Series Name"] = export_import["Series Name"].apply(lambda x: "Import" if "Import" in x else "Export")
    export_import.reset_index(inplace=True, drop=True)
    export_import.set_index(["Country", "Series Name"], inplace=True, drop=True)
    for ind in range(2020, 2025):
        export_import[str(ind)] = export_import[str(ind)].map(convert_to_float_if_possible)
    date_range = pd.date_range(start=f"2020-01-01", end=f"2023-12-31", freq='D')
    data = DataFrame({f"{ind[0]} {ind[1]}": generate_data(export_import.loc[ind, export_import.loc[ind].notna().tolist()], date_range.shape[0]) for ind in export_import.index}, index=date_range)
    return data


def get_gdp() -> DataFrame:
    gdp = pd.read_csv('datasets/gdp_2020_2025.csv')
    gdp = gdp.set_index("Country")
    for ind in range(2020, 2026):
        gdp[str(ind)] = gdp[str(ind)].map(convert_to_float_if_possible)
    date_range = pd.date_range(start=f"2020-01-01", end=f"2024-12-31", freq='D')
    data = DataFrame({country: generate_data(gdp.loc[country, gdp.loc[country].notna().tolist()], date_range.shape[0]) for country in gdp.index}, index=date_range)
    return data


def get_exchange_rates() -> DataFrame:
    exchange_rates = pd.read_csv('datasets/exchange_rates_2021_2025.csv')
    exchange_rates.drop("Unnamed: 0", axis=1, inplace=True)
    exchange_rates["Country"] = exchange_rates["Country"].apply(lambda x: " ".join(x.split()[:-1]))
    exchange_rates["date"] = pd.to_datetime(exchange_rates["date"], format="%d/%m/%Y")
    countries = exchange_rates["Country"].unique()
    date_range = exchange_rates["date"].unique()
    data = DataFrame({country: get_country_exchange_rates(exchange_rates[exchange_rates["Country"] == country], date_range.shape[0]) for country in countries}, index=date_range)
    return data


def bound_df_by_index(df: DataFrame, bound_from, bound_to) -> DataFrame:
    return df.loc[bound_from:bound_to]


def get_min_index(df: DataFrame):
    return min(df.index.tolist())


def get_max_index(df: DataFrame):
    return max(df.index.tolist())


def dataset_prepare():
    export_import = get_export_import()
    gdp = get_gdp()
    exchange_rates = get_exchange_rates()

    min_date = max(get_min_index(export_import), get_min_index(gdp), get_min_index(exchange_rates))
    max_date = min(get_max_index(export_import), get_max_index(gdp), get_max_index(exchange_rates))

    export_import = bound_df_by_index(export_import, min_date, max_date)
    gdp = bound_df_by_index(gdp, min_date, max_date)
    exchange_rates = bound_df_by_index(exchange_rates, min_date, max_date)

    countries = list(filter(lambda x: x.strip() and any(x in s for s in export_import.columns) and any(x in s for s in gdp.columns), exchange_rates.columns.tolist()))

    for country in countries:
        df = DataFrame(index=exchange_rates.index)
        df['Rate'] = exchange_rates[country]
        df['Gdp'] = gdp[country]
        df['Import'] = export_import[f"{country} Import"]
        df['Export'] = export_import[f"{country} Export"]
        df.to_csv(f"datasets/by_country/{country}.csv")



dataset_prepare()
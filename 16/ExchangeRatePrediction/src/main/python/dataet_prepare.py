import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pandas import DataFrame
from numpy import arange, array, nan

from config import settings
from util import LogProgressInfo

def get_country_exchange_rates(df: DataFrame, target_size, progress_info: LogProgressInfo = None):
    if progress_info is not None:
        progress_info.print()

    data = df.groupby('date').max()['value']
    return [*data, *([nan] * (target_size - data.shape[0]))]


def generate_linear_data_for_period(year_from: int, year_to: int, data_sum: float) -> list[float]:
    total_days = (pd.to_datetime(f'{year_to}-01-01')
                  - pd.to_datetime(f'{year_from}-01-01')).days
    k = data_sum / (total_days * total_days)
    result = [k * i for i in range(total_days)]
    return result


def generate_linear_data(years: list[int], data: list[float]) -> array:
    return array(sum([generate_linear_data_for_period(years[i], years[i + 1], data[i]) for i in range(len(years) - 1)], []))


def convert_to_float_if_possible(s: str):
    try:
        return float(s)
    except ValueError:
        return nan


def generate_data(df, target_size, progress_info: LogProgressInfo = None):
    if progress_info is not None:
        progress_info.print()
    years = list(map(int, df.index.tolist()))
    if len(years) == 0:
        return [nan] * target_size
    years.append(years[-1] + 1)
    generated = generate_linear_data(years, df.tolist())
    if len(generated) < target_size:
        generated = [*generated, *([nan] * (target_size - generated.shape[0]))]
    elif len(generated) > target_size:
        generated= generated[:target_size]
    return generated


def get_export_import() -> DataFrame:
    export_import = pd.read_csv(settings.import_export_filepath)
    keys = ["Country", "Series Name"] + [str(i) for i in range(settings.import_export_year_from, settings.import_export_year_to + 1)]
    export_import = export_import[keys]
    export_import = export_import[export_import["Series Name"].isin(['Imports of goods and services (current US$)', 'Exports of goods and services (BoP, current US$)'])]
    export_import["Series Name"] = export_import["Series Name"].apply(lambda x: "Import" if "Import" in x else "Export")
    export_import.reset_index(inplace=True, drop=True)
    export_import.set_index(["Country", "Series Name"], inplace=True, drop=True)
    for ind in range(settings.import_export_year_from, settings.import_export_year_to + 1):
        export_import[str(ind)] = export_import[str(ind)].map(convert_to_float_if_possible)
    date_range = pd.date_range(start=f"{settings.import_export_year_from + 1}-01-01", end=f"{settings.import_export_year_to + 1}-12-31", freq='D')

    data = DataFrame({f"{ind[0]} {ind[1]}": generate_data(
        export_import.loc[ind,export_import.loc[ind].notna().tolist()],
        date_range.shape[0],
        LogProgressInfo("Export/Import", progress, len(export_import.index)))
        for ind, progress in zip(export_import.index, range(1, len(export_import.index) + 1))}, index=date_range)

    print()
    return data

def get_gdp() -> DataFrame:
    gdp = pd.read_csv(settings.gdp_filepath)
    gdp = gdp.set_index("Country")
    for ind in range(settings.gdp_year_from, settings.gdp_year_to + 1):
        gdp[str(ind)] = gdp[str(ind)].map(convert_to_float_if_possible)
    date_range = pd.date_range(start=f"{settings.gdp_year_from + 1}-01-01", end=f"{settings.gdp_year_to + 1}-12-31", freq='D')
    data = DataFrame({country: generate_data(
        gdp.loc[country, gdp.loc[country].notna().tolist()],
        date_range.shape[0],
        LogProgressInfo("GDP", progress, len(gdp.index)))
        for country, progress in zip(gdp.index, range(1, len(gdp.index) + 1))}, index=date_range)

    print()
    return data

def get_exchange_rates() -> DataFrame:
    exchange_rates = pd.read_csv(settings.currency_filepath)
    exchange_rates.drop("Unnamed: 0", axis=1, inplace=True)
    exchange_rates["Country"] = exchange_rates["Country"].apply(lambda x: " ".join(x.split()[:-1]))
    exchange_rates["date"] = pd.to_datetime(exchange_rates["date"], format="%d/%m/%Y")
    countries = exchange_rates["Country"].unique()
    date_range = exchange_rates["date"].unique()
    data = DataFrame({country: get_country_exchange_rates(
        exchange_rates[exchange_rates["Country"] == country],
        date_range.shape[0],
        LogProgressInfo("Exchange rates", progress, len(countries)))
        for country, progress in zip(countries, range(1, len(countries) + 1))}, index=date_range)

    print()
    return data


def bound_df_by_index(df: DataFrame, bound_from, bound_to) -> DataFrame:
    return df.loc[bound_from:bound_to]


def get_min_index(df: DataFrame):
    return min(df.index.tolist())


def get_max_index(df: DataFrame):
    return max(df.index.tolist())


def dataset_prepare():
    export_import = get_export_import()
    print("Export/import loaded")
    gdp = get_gdp()
    print("GDP loaded")
    exchange_rates = get_exchange_rates()
    print("Exchange rates loaded")

    min_date = max(get_min_index(export_import), get_min_index(gdp), get_min_index(exchange_rates))
    max_date = min(get_max_index(export_import), get_max_index(gdp), get_max_index(exchange_rates))

    export_import = bound_df_by_index(export_import, min_date, max_date)
    gdp = bound_df_by_index(gdp, min_date, max_date)
    exchange_rates = bound_df_by_index(exchange_rates, min_date, max_date)

    countries = list(filter(lambda x: x.strip()
                                      and any(x in s for s in export_import.columns)
                                      and any(x in s for s in gdp.columns), exchange_rates.columns.tolist()))

    for country, progress in zip(countries, range(1, len(countries) + 1)):
        df = DataFrame(index=exchange_rates.index)
        df['Rate'] = exchange_rates[country]
        df['Gdp'] = gdp[country]
        df['Import'] = export_import[f"{country} Import"]
        df['Export'] = export_import[f"{country} Export"]
        df.to_csv(f"{settings.prepared_data_dir}{country}{settings.file_format}")
        LogProgressInfo("Prepare data", progress, len(countries))

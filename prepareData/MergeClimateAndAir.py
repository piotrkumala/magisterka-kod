import pandas as pd

from prepareData import prepare_air_data, prepare_climate_data


def merge_climate_and_air_data(cities: list[dict[str, int | str]]):
    result = []

    for city in cities:
        result.append(
            prepare_climate_data(city['stationCode']).merge(prepare_air_data(city['cityName'])[['pm10', 'date']],
                                                            on='date'))
    return pd.concat(result)

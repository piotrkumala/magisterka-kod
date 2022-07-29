from prepareData import prepare_air_data, prepare_climate_data


def merge_climate_and_air_data():
    air = prepare_air_data()
    climate = prepare_climate_data()

    return climate.merge(air[['pm10', 'date']], on='date')
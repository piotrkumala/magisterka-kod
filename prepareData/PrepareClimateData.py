import pandas as pd
from matplotlib import pyplot as plt


def prepare_climate_data(stationCode: int):
    d_names = ['station_code', 'station_name', 'year', 'month', 'day', 'max_temp', 'status_max_temp', 'min_temp',
               'status_min_temp', 'mean_temp', 'status_mean_temp', 'min_ground_temp', 'status_min_ground_temp',
               'sum_precipitation', 'status_sum_precipitation', 'type_precipitation', 'show_height',
               'status_snow_height']
    d = pd.concat([
        pd.concat([
            pd.read_csv(f'data/climate/{year}/k_d_{str(month).zfill(2)}_{year}.csv', names=d_names,
                        encoding='iso-8859-1')
            for month in range(1, 13)
        ]) for year in range(2001, 2021)])

    d['month'] = d['month'].apply(lambda x: str(x).zfill(2))
    d['day'] = d['day'].apply(lambda x: str(x).zfill(2))

    d = d.assign(date=d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str))
    d['date'] = pd.to_datetime(d['date'])

    d_local = d.loc[d['station_code'] == stationCode]

    plt.figure(figsize=(18, 8))
    plt.scatter(
        d_local['date'],
        d_local['mean_temp'])
    plt.title(f'Mean 24-hour temperature in Krakow from 2001 to 2020')
    plt.xlabel('Date')
    plt.ylabel('Temperature [$^\circ$C]')
    plt.legend()
    plt.show()
    return d_local[
        ['date', 'max_temp', 'min_temp', 'mean_temp', 'min_ground_temp', 'sum_precipitation', 'station_code',
         'station_name', 'type_precipitation', 'show_height']]

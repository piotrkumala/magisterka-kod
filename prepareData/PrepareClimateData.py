import pandas as pd
import matplotlib.pyplot as plt


def prepare_climate_data():
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

    d_krakow = d.loc[d['station_code'] == 250190390]
    # plt.figure(figsize=(24, 12))
    # plt.plot_date(d_krakow['date'].values, d_krakow['mean_temp'].values)
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.show()
    return d_krakow[
        ['date', 'max_temp', 'min_temp', 'mean_temp', 'min_ground_temp', 'sum_precipitation', 'station_code',
         'station_name', 'type_precipitation', 'show_height']]

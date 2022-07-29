import pandas as pd
import matplotlib.pyplot as plt


def prepare_air_data():
    df = pd.concat([
        *[pd.read_excel(f'data/air/{year}/{year}_PM10_24g.xlsx', skiprows=[1, 2]) for year in
          range(2001, 2016)],
        *[pd.read_excel(f'data/air/{year}/{year}_PM10_24g.xlsx', skiprows=[1, 2, 3, 4]) for year in range(2016, 2021)]
    ])

    krakow_cols = [cols for cols in df.columns if 'MpKrak' in cols]
    # plt.figure(figsize=(24, 12))
    # plt.plot_date(df['Kod stacji'].values, df[krakow_cols].values)
    # plt.xticks(rotation=45)
    # plt.legend(krakow_cols)
    # plt.show()
    df[krakow_cols] = df[krakow_cols].apply(pd.to_numeric, errors='coerce', axis=1)
    df['pm10'] = df[krakow_cols].mean(axis=1);
    df['date'] = pd.to_datetime(df['Kod stacji'])
    return df[['date', 'pm10', *krakow_cols]]

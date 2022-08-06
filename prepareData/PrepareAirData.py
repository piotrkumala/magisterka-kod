import pandas as pd


def prepare_air_data(city_name: str):
    df = pd.concat([
        *[pd.read_excel(f'data/air/{year}/{year}_PM10_24g.xlsx', skiprows=[1, 2]) for year in
          range(2001, 2016)],
        *[pd.read_excel(f'data/air/{year}/{year}_PM10_24g.xlsx', skiprows=[1, 2, 3, 4]) for year in range(2016, 2021)]
    ])

    city_cols = [cols for cols in df.columns if city_name in cols]
    df[city_cols] = df[city_cols].apply(pd.to_numeric, errors='coerce', axis=1)
    df['pm10'] = df[city_cols].mean(axis=1)
    df['date'] = pd.to_datetime(df['Kod stacji'])
    return df[['date', 'pm10', *city_cols]]

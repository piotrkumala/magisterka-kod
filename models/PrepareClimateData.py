import pandas as pd


def prepare_climate_data(data_path: str) -> [pd.Series, pd.DataFrame]:
    df = pd.read_csv(data_path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(['station_code', 'station_name', 'type_precipitation', 'show_height'], axis=1)
    df = df.fillna(0)
    # df = df[df['date'].apply(lambda x: x.year == 2013)]

    x: pd.Series = df[[col for col in df.columns if col not in ['pm10']]]
    y: pd.DataFrame = df['pm10']
    return x, y

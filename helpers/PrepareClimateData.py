import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_climate_data(df: pd.DataFrame) -> [pd.Series, pd.DataFrame]:
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(['station_code', 'station_name', 'type_precipitation', 'show_height'], axis=1)
    df = df.fillna(0)
    # df = df[df['date'].apply(lambda x: x.year == 2013)]

    x: pd.Series = df[[col for col in df.columns if col not in ['pm10']]]
    y: pd.DataFrame = df['pm10']
    return x, y


def prepare_climate_data_for_rnn(df: pd.DataFrame) -> [pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    x, y = prepare_climate_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    date = x_test['date']
    x_train = x_train.loc[:, x_train.columns != 'date'].to_numpy()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).swapaxes(1, 2)
    x_test = x_test.loc[:, x_test.columns != 'date'].to_numpy()
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).swapaxes(1, 2)

    return x_train, y_train, x_test, y_test, date

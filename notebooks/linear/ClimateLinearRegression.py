import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import helpers


def run_linear_regression_for_climate(df: pd.DataFrame, plots_directories: str):
    x, y = helpers.prepare_climate_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train.loc[:, x_train.columns != 'date'], y_train)

    y_prediction = lr.predict(x_test.loc[:, x_test.columns != 'date'])

    helpers.plot_predicted_and_real_values(x_test['date'], y_test, y_prediction, 'linear', plots_directories)

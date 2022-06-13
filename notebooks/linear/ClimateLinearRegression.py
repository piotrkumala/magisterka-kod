import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import helpers


def run_linear_regression_for_climate(data_path: str):
    x, y = helpers.prepare_climate_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train.loc[:, x_train.columns != 'date'], y_train)

    y_prediction = lr.predict(x_test.loc[:, x_test.columns != 'date'])

    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqrd_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared  error  of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))

    helpers.plot_predicted_and_real_values(x_test['date'], y_test, y_prediction, 'linear')


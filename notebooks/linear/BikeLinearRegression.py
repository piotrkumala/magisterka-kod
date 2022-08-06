import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def run_linear_regression_for_bikes(data_path: str):
    df = pd.read_csv(data_path)
    # df['date'] = pd.to_datetime(df['Date'])

    # df['Road_conditions_Snow'].plot()

    x = df[[col for col in df.columns if col not in ['Accident_Count']]]
    y = df['Accident_Count']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train.loc[:, x_train.columns != 'Date'], y_train)

    y_prediction = lr.predict(x_test.loc[:, x_test.columns != 'Date'])

    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqr_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared  error  of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))

    plt.title('Total accidents per day')
    plt.plot(x_test['Date'], y_test)
    # plt.scatter(x_test['Date'], y_prediction)
    plt.legend()
    plt.show()

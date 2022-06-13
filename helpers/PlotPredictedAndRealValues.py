import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def plot_predicted_and_real_values(date, y_test, y_prediction, plot_title):
    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqrt_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared  error  of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))
    plt.figure(figsize=(18, 8))
    plt.scatter(date, y_test)
    plt.scatter(date, y_prediction)
    plt.legend(['real pm10 level', 'predicted pm10 level'])
    plt.xlabel('Date')
    plt.ylabel('PM10 [ug/m^3]')
    plt.title(f'PM10 level (real and predicted) in Cracow from 2000 to 2021 using {plot_title} regression')
    plt.show()

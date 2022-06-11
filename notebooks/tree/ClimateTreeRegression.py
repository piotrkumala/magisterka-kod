import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from models.PrepareClimateData import prepare_climate_data


def climate_tree_regression(data_path: str):
    x, y = prepare_climate_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    dtr = DecisionTreeRegressor(max_depth=3)

    dtr.fit(x_train.loc[:, x_train.columns != 'date'], y_train)

    y_prediction = dtr.predict(x_test.loc[:, x_test.columns != 'date'])
    print(x_test.loc[:, x_test.columns != 'date'].columns.values)

    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqrt_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared  error  of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))

    tree.export_graphviz(dtr, feature_names=x_test.loc[:, x_test.columns != 'date'].columns.values, out_file='tree.dot')
    plt.figure(figsize=(18, 8))
    plt.scatter(x_test['date'], y_test)
    plt.scatter(x_test['date'], y_prediction)
    plt.legend(['real pm10 level', 'predicted pm10 level'])
    plt.xlabel('Date')
    plt.ylabel('PM10 [ug/m^3]')
    plt.title('PM10 level (real and predicted) in Cracow from 2000 to 2021 using decision tree '
              'regression')
    plt.show()
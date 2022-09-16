import numpy as np
import pandas as pd

import helpers
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def climate_tree_regression(df: pd.DataFrame, plots_directories: str, tree_depth: int) -> object:
    x, y = helpers.prepare_climate_data(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    dtr = DecisionTreeRegressor(max_depth=tree_depth)

    dtr.fit(x_train.loc[:, x_train.columns != 'date'], y_train)

    y_prediction = dtr.predict(x_test.loc[:, x_test.columns != 'date'])
    print(x_test.loc[:, x_test.columns != 'date'].columns.values)

    score = r2_score(y_test, y_prediction)
    print('r2 score is ', score)
    print('mean_sqrt_error is ==', mean_squared_error(y_test, y_prediction))
    print('root_mean_squared  error  of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))

    tree.export_graphviz(dtr, feature_names=x_test.loc[:, x_test.columns != 'date'].columns.values, out_file='tree.dot')
    helpers.plot_predicted_and_real_values(x_test['date'], y_test, y_prediction, 'decision tree',
                                           plots_directories, tree_depth)
    return score

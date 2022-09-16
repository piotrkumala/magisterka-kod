import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

import helpers
import notebooks
import prepareData


def main():
    df = prepareData.merge_climate_and_air_data([{"stationCode": 250190390, "cityName": 'MpKrak'}])

    # notebooks.climate_tree_regression(df.copy(), 'models', 5)
    # test_scalability()
    # compare_models(df)
    # compare_trees(df)

    # helper_for_plotting()


def helper_for_plotting():
    plt.figure(figsize=(18, 8))
    plt.plot(
        range(3, 15),
        [0.31, 0.33, 0.28, 0.25, 0.20, 0.13, 0.05, -0.01, -0.12, -0.13, -0.19, -0.28])
    plt.title(f'R^2 score for different tree depth')
    plt.xlabel('Tree depth')
    plt.ylabel('R^2 score')
    plt.legend()
    plt.show()


def compare_trees(df):
    history = []
    for i in range(3, 15):
        history.append(notebooks.climate_tree_regression(df.copy(), 'models', i))
    print(history)
    helpers.plot_for_trees(range(3, 15), history)


def test_scalability():
    dictionaries_array = [{"stationCode": 250190390, "cityName": 'MpKrak'},
                          {"stationCode": 252200150, "cityName": 'MzWar'},
                          {"stationCode": 349220695, "cityName": 'PkPrzem'},
                          {"stationCode": 249180010, "cityName": 'SlPszcz'},
                          {"stationCode": 249200180, "cityName": 'MpLimano'},
                          {"stationCode": 249220080, "cityName": 'PkSano'},
                          {"stationCode": 250140030, "cityName": 'DsBoga'},
                          {"stationCode": 250160130, "cityName": 'DsSzcz'},
                          {"stationCode": 250160360, "cityName": 'MpTar'},
                          {"stationCode": 250170390, "cityName": 'OpGlub'},
                          {"stationCode": 250180030, "cityName": 'OpOles'},
                          {"stationCode": 250190250, "cityName": 'DsZabk'},
                          {"stationCode": 250190530, "cityName": 'SlKato'},
                          {"stationCode": 250210050, "cityName": 'SkSw'},
                          {"stationCode": 251150250, "cityName": 'DsZgor'},
                          {"stationCode": 251160150, "cityName": 'DsPol'},
                          {"stationCode": 251180090, "cityName": 'LdSier'},
                          {"stationCode": 251200030, "cityName": 'LdSkier'},
                          {"stationCode": 251210040, "cityName": 'LbJarcz'},
                          {"stationCode": 251210120, "cityName": 'LbPula'},
                          {"stationCode": 252170110, "cityName": 'WpGnie'},
                          {"stationCode": 252200120, "cityName": 'MzLeg'},
                          {"stationCode": 253180150, "cityName": 'KpGrud'},
                          {"stationCode": 253180220, "cityName": 'KpBydg'},
                          {"stationCode": 254170140, "cityName": 'PmKos'},
                          {"stationCode": 254180060, "cityName": 'PmGdy'},
                          {"stationCode": 254180110, "cityName": 'PmGda'},
                          {"stationCode": 254220030, "cityName": 'WmGold'},
                          {"stationCode": 249180130, "cityName": 'SlCies'}]
    history_list = []
    df = pandas.DataFrame([])
    for i in range(1, len(dictionaries_array) + 1):
        arr = [dictionaries_array[i - 1]]
        df = pd.concat([df, prepareData.merge_climate_and_air_data(arr)])
        df = df[df['pm10'].notna()]
        gru = notebooks.gru_regression(df.copy(), 32, 50, 'scale_new', i)
        history_list.append(gru)
    helpers.plot_willmott_for_cities(range(1, len(dictionaries_array) + 1), history_list)


def compare_models(df):
    notebooks.run_linear_regression_for_climate(df.copy(), 'models')
    notebooks.climate_tree_regression(df.copy(), 'models', 5)
    history_list = []
    for neurons in [256]:
        rnn = notebooks.simple_rnn_regression(df.copy(), neurons, 50, 'models')
        lstm = notebooks.lstm_regression(df.copy(), neurons, 50, 'models')
        gru = notebooks.gru_regression(df.copy(), neurons, 50, 'models')

        rnn.label = f'Simple RNN with {neurons} neurons'
        lstm.label = f'LSTM with {neurons} neurons'
        gru.label = f'GRU with {neurons} neurons'
        history_list.extend([rnn, lstm, gru])
    helpers.plot_model_history(history_list)


main()

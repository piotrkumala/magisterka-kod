import helpers
import notebooks
import prepareData


def main():
    dictionaries_array = [{"stationCode": 250190390, "cityName": 'MpKrak'}, {"stationCode": 252200150, "cityName": 'MzWar'}, {"stationCode": 349220695, "cityName": 'PkPrzem'}, {"stationCode": 249180010, "cityName": 'SlPszcz'}]
    history_list = []
    for i in range(1, len(dictionaries_array) + 1):
        arr = dictionaries_array[slice(0, i)]
        df = prepareData.merge_climate_and_air_data(arr)
        gru = notebooks.gru_regression(df.copy(), 128, 50, 'scale', i)
        history_list.append(gru)
    helpers.plot_willmott_for_cities(range(1, len(dictionaries_array) + 1), history_list)

    compare_models(df)


def compare_models(df):
    notebooks.run_linear_regression_for_climate(df.copy(), 'models')
    notebooks.climate_tree_regression(df.copy(), 'models')
    history_list = []
    for neurons in [32, 64, 128, 216, 512, 1024]:
        rnn = notebooks.simple_rnn_regression(df.copy(), neurons, 50, 'models')
        lstm = notebooks.lstm_regression(df.copy(), neurons, 50, 'models')
        gru = notebooks.gru_regression(df.copy(), neurons, 50, 'models')

        rnn.label = f'Simple RNN with {neurons} neurons'
        lstm.label = f'LSTM with {neurons} neurons'
        gru.label = f'GRU with {neurons} neurons'
        history_list.extend([rnn, lstm, gru])
    helpers.plot_model_history(history_list)


main()

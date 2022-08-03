import helpers
import notebooks
import prepareData


def main():
    df = prepareData.merge_climate_and_air_data(
        [{"stationCode": 250190390, "cityName": 'MpKrak'}, {"stationCode": 252200150, "cityName": 'MzWar'}])
    gru = notebooks.gru_regression(df.copy(), 128, 50)

    # compare_models(df)


def compare_models(df):
    notebooks.run_linear_regression_for_climate(df.copy())
    notebooks.climate_tree_regression(df.copy())
    history_list = []
    for neurons in [32, 64, 128, 216, 512, 1024]:
        rnn = notebooks.simple_rnn_regression(df.copy(), neurons, 50)
        lstm = notebooks.lstm_regression(df.copy(), neurons, 50)
        gru = notebooks.gru_regression(df.copy(), neurons, 50)

        rnn.label = f'Simple RNN with {neurons} neurons'
        lstm.label = f'LSTM with {neurons} neurons'
        gru.label = f'GRU with {neurons} neurons'
        history_list.extend([rnn, lstm, gru])
    helpers.plot_model_history(history_list)


main()

import helpers
import notebooks
import prepareData

df = prepareData.merge_climate_and_air_data()
# notebooks.run_linear_regression_for_bikes('./data/Accidents_done.csv')
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

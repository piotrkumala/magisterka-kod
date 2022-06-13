import helpers
from notebooks.gru.GRURegression import gru_regression

from notebooks.linear.ClimateLinearRegression import run_linear_regression_for_climate
from notebooks.lstm.LSTMRegression import lstm_regression
from notebooks.simple_rnn.SimpleRNNRegression import simple_rnn_regression
from notebooks.tree.ClimateTreeRegression import climate_tree_regression

# run_linear_regression_for_bikes('./data/Accidents_done.csv')
run_linear_regression_for_climate('./data/climate_and_air.csv')

climate_tree_regression('data/climate_and_air.csv')

history_list = []

for neurons in [32, 64, 128, 216, 512, 1024]:
    rnn = simple_rnn_regression('data/climate_and_air.csv', neurons, 50)
    lstm = lstm_regression('data/climate_and_air.csv', neurons, 50)
    gru = gru_regression('data/climate_and_air.csv', neurons, 50)

    rnn.label = f'Simple RNN with {neurons} neurons'
    lstm.label = f'LSTM with {neurons} neurons'
    gru.label = f'GRU with {neurons} neurons'
    history_list.extend([rnn, lstm, gru])


helpers.plot_model_history(history_list)

import helpers

from notebooks.linear.ClimateLinearRegression import run_linear_regression_for_climate
from notebooks.lstm.LSTMRegression import lstm_regression
from notebooks.simple_rnn.SimpleRNNRegression import simple_rnn_regression
from notebooks.tree.ClimateTreeRegression import climate_tree_regression

# run_linear_regression_for_bikes('./data/Accidents_done.csv')
run_linear_regression_for_climate('./data/climate_and_air.csv')

climate_tree_regression('data/climate_and_air.csv')

rnn_256_history = simple_rnn_regression('data/climate_and_air.csv', 256, 50)
rnn_256_history.label = 'Simple RNN with 256 neurons'

rnn_128_history = simple_rnn_regression('data/climate_and_air.csv', 128, 50)
rnn_128_history.label = 'Simple RNN with 128 neurons'

rnn_64_history = simple_rnn_regression('data/climate_and_air.csv', 64, 50)
rnn_64_history.label = 'Simple RNN with 64 neurons'

rnn_32_history = simple_rnn_regression('data/climate_and_air.csv', 32, 50)
rnn_32_history.label = 'Simple RNN with 32 neurons'

lstm_256_history = lstm_regression('data/climate_and_air.csv', 256, 50)
lstm_256_history.label = 'LSTM with 256 neurons'

lstm_128_history = lstm_regression('data/climate_and_air.csv', 128, 50)
lstm_128_history.label = 'LSTM with 128 neurons'

lstm_64_history = lstm_regression('data/climate_and_air.csv', 64, 50)
lstm_64_history.label = 'LSTM with 64 neurons'

lstm_32_history = lstm_regression('data/climate_and_air.csv', 32, 50)
lstm_32_history.label = 'LSTM with 32 neurons'

history_list = [rnn_256_history, rnn_128_history, rnn_64_history, rnn_32_history,
                lstm_256_history, lstm_128_history, lstm_64_history, lstm_32_history]
helpers.plot_model_history(history_list)

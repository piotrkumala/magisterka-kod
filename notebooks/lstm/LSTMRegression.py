import pandas as pd
import tensorflow as tf
import helpers


def lstm_regression(df: pd.DataFrame, neurons: int, epochs: int, plots_directories: str):
    x_train, y_train, x_test, y_test, date = helpers.prepare_climate_data_for_rnn(df)

    model: tf.keras.Model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(neurons))

    model.add(tf.keras.layers.Dense(1))

    history, y_prediction = helpers.perform_rnn_regression(model, x_test, x_train, y_train, epochs)

    helpers.plot_predicted_and_real_values(date, y_test, y_prediction,
                                           'long-short term memory neural network', plots_directories, neurons)

    return history

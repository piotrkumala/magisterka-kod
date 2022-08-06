import pandas as pd
import tensorflow as tf
import helpers as helpers


def gru_regression(df: pd.DataFrame, neurons: int, epochs: int, plots_directories: str, number_of_cities: int = 0):
    x_train, y_train, x_test, y_test, date = helpers.PrepareClimateData.prepare_climate_data_for_rnn(df)

    model: tf.keras.Model = tf.keras.Sequential()

    model.add(tf.keras.layers.GRU(neurons))

    model.add(tf.keras.layers.Dense(1))

    history, y_prediction = helpers.perform_rnn_regression(model, x_test, x_train, y_train, epochs)

    willmott_index = helpers.plot_predicted_and_real_values(date, y_test, y_prediction,
                                                            'GRU neural network', plots_directories, neurons,
                                                            number_of_cities)

    return history if number_of_cities == 0 else willmott_index

import tensorflow as tf
import helpers as helpers


def gru_regression(data_path: str, neurons: int, epochs: int):
    x_train, y_train, x_test, y_test, date = helpers.PrepareClimateData.prepare_climate_data_for_rnn(data_path)

    model: tf.keras.Model = tf.keras.Sequential()

    model.add(tf.keras.layers.GRU(neurons))

    model.add(tf.keras.layers.Dense(1))

    history, y_prediction = helpers.perform_rnn_regression(model, x_test, x_train, y_train, epochs)

    helpers.plot_predicted_and_real_values(date, y_test, y_prediction,
                                           'GRU neural network')

    return history

import tensorflow as tf
from keras import Model

from models.PlotModelHistory import plot_model_history
from models.PlotPredictedAndRealValues import plot_predicted_and_real_values
from models.PrepareClimateData import prepare_climate_data, prepare_climate_data_for_rnn
from tensorflow import keras
from tensorflow.keras import layers


def simple_rnn_regression(data_path: str):
    x_train, y_train, x_test, y_test, date = prepare_climate_data_for_rnn(data_path)

    model: Model = keras.Sequential()

    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(1))

    model.build(x_test.shape)
    model.summary()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(x_train, y_train, epochs=50)
    y_prediction = model.predict(x_test)

    plot_predicted_and_real_values(date, y_test, y_prediction, 'simple recurrent neural network')
    plot_model_history(history, 'Simple RNN')




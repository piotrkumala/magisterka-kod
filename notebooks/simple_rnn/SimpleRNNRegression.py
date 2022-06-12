import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from models.PlotPredictedAndRealValues import plot_predicted_and_real_values
from models.PrepareClimateData import prepare_climate_data
from tensorflow import keras
from tensorflow.keras import layers


def simple_rnn_regression(data_path: str):
    x, y = prepare_climate_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    date = x_test['date']
    x_train = x_train.loc[:, x_train.columns != 'date'].to_numpy()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).swapaxes(1, 2)
    x_test = x_test.loc[:, x_test.columns != 'date'].to_numpy()
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).swapaxes(1, 2)

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

    plot_predicted_and_real_values(y_test, y_prediction, date)




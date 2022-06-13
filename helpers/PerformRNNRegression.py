import tensorflow as tf


def perform_rnn_regression(model, x_test, x_train, y_train, epochs):
    model.build(x_test.shape)
    model.summary()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(x_train, y_train, epochs=epochs)
    y_prediction = model.predict(x_test)
    return history, y_prediction

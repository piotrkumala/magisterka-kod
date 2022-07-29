import tensorflow as tf


def perform_rnn_regression(model, x_test, x_train, y_train, epochs):
    print(x_test.shape)
    print(x_train.shape)

    model.build(x_test.shape)
    model.summary()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    y_prediction = model.predict(x_test)
    return history, y_prediction

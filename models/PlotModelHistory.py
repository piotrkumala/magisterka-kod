import matplotlib.pyplot as plt


def plot_model_history(history, model_name):
    plt.figure(figsize=(18, 8))
    plt.plot(history.history['loss'], label='loss')
    plt.title(f'{model_name} performance')
    plt.legend()
    plt.show()
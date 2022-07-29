import matplotlib.pyplot as plt


def plot_model_history(history_list):
    plt.figure(figsize=(18, 8))
    for item in history_list:
        plt.plot(item.history['loss'], label=item.label)
    plt.title(f'Loss function of RNN models')
    plt.legend()
    plt.savefig('img/loss_history.png')

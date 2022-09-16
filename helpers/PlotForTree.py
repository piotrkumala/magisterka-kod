from matplotlib import pyplot as plt


def plot_for_trees(x: range, y: list[float]) -> None:
    fig = plt.figure(figsize=(18, 8))
    plt.plot(x, y)
    plt.title(f'R^2 score for different tree depth')
    plt.savefig('img/models/trees_willmott.png')
    plt.close(fig)
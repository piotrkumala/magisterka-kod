from matplotlib import pyplot as plt


def plot_willmott_for_cities(x: range, y: list[float]) -> None:
    fig  = plt.figure(figsize=(18, 8))
    plt.plot(x, y)
    plt.title(f'Willmott indexes for different number of cities')
    plt.savefig('img/scale_new/cities_willmott.png')
    plt.close(fig)

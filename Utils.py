import matplotlib.pyplot as plt


def plotLoss(losses_dict, path):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), ["IC", "BC", "PDE"]):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)


import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_train_val_loss(path: str, label: str, show: bool = False, ax = None, plot_time = False) -> None:
    """
    Plot train and val losses over epoch and time, given path to pickled losses.
    """
    filename = os.path.join(path, 'losses.pickle')
    with open(filename, 'rb') as file:
        d = pickle.load(file)

    epoch_num = np.arange(len(d))
    train_loss = d['loss']
    val_loss = d['val_loss']
    min_val_loss = []
    min_val_loss.append(val_loss[0])
    for i in range(len(val_loss) - 1):
        if val_loss[i + 1] < min_val_loss[-1] * 1.025:
            min_val_loss.append(val_loss[i + 1])
        else:
            min_val_loss.append(min_val_loss[-1])
    start = 2
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=False)

    times = d['time'] / 60.
    times = times - times[0]

    ax.plot(epoch_num[start:], train_loss[start:], '-o', label=label+'train loss', markersize=5)
    ax.plot(epoch_num[start:], min_val_loss[start:], '-^', label=label+'val loss', markersize=5)
    ax.set_xlabel('Training epoch')
    ax.grid(which='major')
    ax.set_ylabel('Loss')

    if plot_time:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = np.arange(0, epoch_num[-1], 50)
        new_tick_locations = list(new_tick_locations)
        new_tick_locations.append(epoch_num[-1] + 1)
        print(new_tick_locations)
        ax2.set_xticks(new_tick_locations)


        def tick_function(X):
            X = np.asarray(X)
            X[X > len(times) - 1] = len(times) - 1
            return np.asarray(times[X], dtype=int)


        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(r"time [minutes]")

    ax.legend()
    if show:
        plt.show()

fig, ax = plt.subplots(constrained_layout=False)
ax.grid(which='major')

path = 'C:\\Users\\Noam\\Desktop\\University\\Research\\MRA\\unrolling_synchronization\\results\\z_2_alignment_mlp_vs_unrolling\\20230727-114642' #MLP
label = 'MLP '
plot_train_val_loss(path,label,show=False,ax=ax)
path = 'C:\\Users\\Noam\\Desktop\\University\\Research\\MRA\\unrolling_synchronization\\results\\z_2_alignment_mlp_vs_unrolling\\20230727-122246' #Unrolled
label = 'Unrolled '
plot_train_val_loss(path,label,show=True,ax=ax)


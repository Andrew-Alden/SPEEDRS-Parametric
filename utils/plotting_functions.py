import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def change_ax_font(tick, fontsize):
    """
    Change axis font
    :param tick: Current set of ticks
    :param fontsize: Fontsize
    :return: Nothing
    """
    try:
        tick.label.set_fontsize(fontsize)
    except:
        try:
            tick.label1.set_fontsize(fontsize)
        except:
            pass


def construct_scatter_plot(ax, x_values, pred, target, labels, markers, colors, alphas, linewidths, xlabel, ylabel,
                           r=3):

    ax.scatter(x_values, pred, label=labels[0], color=colors[0], alpha=alphas[0],
               linewidth=linewidths[0], marker=markers[0])
    ax.scatter(x_values, target, label=labels[1], color=colors[1], alpha=alphas[1],
               linewidth=linewidths[1], marker=markers[1])

    ax.legend(loc='best', fontsize=25)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        change_ax_font(tick, 22)
    for tick in ax.yaxis.get_major_ticks():
        change_ax_font(tick, 22)

    ax.grid(True, color='black', alpha=0.2, linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(f'MSE: {np.round(mean_squared_error(pred, target), r)}', fontsize=25)
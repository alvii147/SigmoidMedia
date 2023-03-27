import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
XLIM = (-5, 5)
YLIM = (-0.5, 1.5)
YTICKS = np.arange(YLIM[0], YLIM[1] + 0.5, 0.5)
RESOLUTION = 1000
COLOR = 'salmon'
LINEWIDTH = 4
SNS_STYLE = {
    'grid.color': '#AAB3C8',
    'axes.facecolor': '#DCDBEE',
    'figure.facecolor' :'none',
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_function(ax, x, f):
    ax.plot(x, f, color=COLOR, linewidth=LINEWIDTH)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_yticks(YTICKS)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    x = np.linspace(*XLIM, RESOLUTION)
    f = sigmoid(x)

    fig, ax = plt.subplots(1)
    plot_function(ax, x, f)
    plt.savefig(SCRIPT_PATH / 'sigmoid.png')

    f = np.heaviside(x, 0.5)

    fig, ax = plt.subplots(1)
    plot_function(ax, x, f)
    plt.savefig(SCRIPT_PATH / 'step.png')

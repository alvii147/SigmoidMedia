import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
THEME = 'light'
XLIM = (-5, 5)
YLIM = (-0.5, 1.5)
YTICKS = np.arange(YLIM[0], YLIM[1] + 0.5, 0.5)
RESOLUTION = 1000
COLOR = '#0A193B' if THEME == 'light' else '#9B67CC'
LINEWIDTH = 3
SNS_STYLE = {
    'grid.color': '#AAB3C8' if THEME == 'light' else '#474072',
    'axes.facecolor': '#DCDBEE' if THEME == 'light' else '#2C2847',
    'figure.facecolor' :'none',
    'xtick.color': '#000000' if THEME == 'light' else '#FFFFFF',
    'ytick.color': '#000000' if THEME == 'light' else '#FFFFFF',
}
SPINE_COLOR = 'black' if THEME == 'light' else 'white'
DPI = 100
ALPHA = 0.8
ZORDER = 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step(x):
    return np.heaviside(x, 0.5)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha):
    return np.maximum(alpha * x, x)


def plot_function(ax, x, f, **kwargs):
    ax.plot(
        x,
        f,
        color=kwargs.get('color', COLOR),
        linewidth=kwargs.get('linewidth', LINEWIDTH),
        alpha=kwargs.get('alpha', ALPHA),
        zorder=ZORDER,
    )

    ax.set_xlim(kwargs.get('xlim', XLIM))
    ax.set_ylim(kwargs.get('ylim', YLIM))
    ax.set_yticks(kwargs.get('yticks', YTICKS))

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    config = {
        'sigmoid': {
            'fn': sigmoid,
            'kwargs': {},
        },
        'step': {
            'fn': step,
            'kwargs': {},
        },
        'tanh': {
            'fn': tanh,
            'kwargs': {
                'ylim': (-1.5, 1.5),
                'yticks': (-1.5, -1, -0.5, 0, 0.5, 1, 1.5),
            },
        },
        'relu': {
            'fn': relu,
            'kwargs': {},
        },
        'leaky_relu': {
            'fn': lambda x: leaky_relu(x, 0.1),
            'kwargs': {},
        },
    }

    x = np.linspace(*XLIM, RESOLUTION)

    for name, cfg in config.items():
        f = cfg['fn'](x)
        fig, ax = plt.subplots()
        plot_function(ax, x, f, **cfg['kwargs'])
        plt.savefig(SCRIPT_PATH / f'{name}_{THEME}.png', dpi=DPI)

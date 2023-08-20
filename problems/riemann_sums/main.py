import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
XLIM = (0, 2)
YLIM = (-0.5, 16.5)
YTICKS = np.arange(YLIM[0], YLIM[1] + 0.5, 4)
RESOLUTION = 1000
FUNCCOLOR = '#0A193B'
FACECOLOR = '#9290F8'
EDGECOLOR = '#3D104B'
SCATTERCOLOR = '#D5664F'
LINEWIDTH = 3
SNS_STYLE = {
    'grid.color': '#AAB3C8',
    'axes.facecolor': '#DCDBEE',
    'figure.facecolor' :'none',
}
DPI = 100
ALPHA = 0.8
ZORDER = 100


def f(x):
    return 16 - np.power(x, 2)


def plot_function(ax, x, rectangle_mode, **kwargs):
    ax.plot(
        x,
        f(x),
        color=kwargs.get('funccolor', FUNCCOLOR),
        linewidth=kwargs.get('linewidth', LINEWIDTH),
        alpha=kwargs.get('alpha', ALPHA),
        zorder=ZORDER,
    )

    xlim = kwargs.get('xlim', XLIM)
    n = kwargs.get('n', 4)
    x_interval = (xlim[1] - xlim[0]) / n

    if rectangle_mode == 'left':
        x_samples = np.linspace(xlim[0], xlim[1], n + 1)[:-1]
        f_samples = f(x_samples)
        for i in range(len(x_samples)):
            ax.add_patch(
                patches.Rectangle(
                    (xlim[0] + (i * x_interval), 0),
                    x_interval,
                    f_samples[i],
                    facecolor=kwargs.get('facecolor', FACECOLOR),
                    edgecolor=kwargs.get('edgecolor', EDGECOLOR),
                    alpha=kwargs.get('alpha', 0.5),
                    linestyle='dashed',
                    linewidth=kwargs.get('linewidth', LINEWIDTH),
                    zorder=ZORDER+1,
                ))
        ax.scatter(
            x_samples,
            f_samples,
            marker='X',
            s=75,
            color=kwargs.get('scattercolor', SCATTERCOLOR),
            zorder=ZORDER+2,
        )
    elif rectangle_mode == 'right':
        x_samples = np.linspace(xlim[0], xlim[1], n + 1)[1:]
        f_samples = f(x_samples)
        for i in range(len(x_samples)):
            ax.add_patch(
                patches.Rectangle(
                    (xlim[0] + (i * x_interval), 0),
                    x_interval,
                    f_samples[i],
                    facecolor=kwargs.get('facecolor', FACECOLOR),
                    edgecolor=kwargs.get('edgecolor', EDGECOLOR),
                    alpha=kwargs.get('alpha', 0.5),
                    linestyle='dashed',
                    linewidth=kwargs.get('linewidth', LINEWIDTH),
                    zorder=ZORDER+1,
                ))
        ax.scatter(
            x_samples,
            f_samples,
            marker='X',
            s=75,
            color=kwargs.get('scattercolor', SCATTERCOLOR),
            zorder=ZORDER+2,
        )
    elif rectangle_mode == 'midpoint':
        x_samples = np.linspace(xlim[0], xlim[1], n + 1)[:-1] + (x_interval * 0.5)
        f_samples = f(x_samples)
        for i in range(len(x_samples)):
            ax.add_patch(
                patches.Rectangle(
                    (xlim[0] + (i * x_interval), 0),
                    x_interval,
                    f_samples[i],
                    facecolor=kwargs.get('facecolor', FACECOLOR),
                    edgecolor=kwargs.get('edgecolor', EDGECOLOR),
                    alpha=kwargs.get('alpha', 0.5),
                    linestyle='dashed',
                    linewidth=kwargs.get('linewidth', LINEWIDTH),
                    zorder=ZORDER+1,
                ))
        ax.scatter(
            x_samples,
            f_samples,
            marker='X',
            s=75,
            color=kwargs.get('scattercolor', SCATTERCOLOR),
            zorder=ZORDER+2,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(kwargs.get('ylim', YLIM))
    ax.set_yticks(kwargs.get('yticks', YTICKS))


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    config = {
        'function': {
            'rectangle_mode': 'none',
            'kwargs': {},
        },
        'left_riemann_sum': {
            'rectangle_mode': 'left',
            'kwargs': {
                'n': 4,
            },
        },
        'right_riemann_sum': {
            'rectangle_mode': 'right',
            'kwargs': {
                'n': 4,
            },
        },
        'midpoint_riemann_sum': {
            'rectangle_mode': 'midpoint',
            'kwargs': {
                'n': 4,
            },
        },
    }

    x = np.linspace(*XLIM, RESOLUTION)

    for name, cfg in config.items():
        fig, ax = plt.subplots()
        plot_function(ax, x, cfg['rectangle_mode'], **cfg['kwargs'])
        plt.savefig(SCRIPT_PATH / f'{name}.png', dpi=DPI)

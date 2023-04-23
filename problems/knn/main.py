import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
XLIM = (-0.5, 4.5)
YLIM = (-0.5, 4.5)
XTICKS = np.arange(XLIM[0], XLIM[1] + 0.5, 0.5)
YTICKS = np.arange(YLIM[0], YLIM[1] + 0.5, 0.5)
CLASS_1_COLOR = '#4800FF'
CLASS_2_COLOR = '#BF333C'
NEW_POINT_COLOR = '#00A058'
ALPHA = 1
NEIGHBOURS_COLOR = '#0A193B'
NEIGHBOURS_ALPHA = 0.5
CLASS_1_MARKER = 'X'
CLASS_2_MARKER = 'o'
NEW_POINT_MARKER = '^'
MARKER_SIZE = 80
NEIGHBOURS_RADIUS_SCALE = 1.2
NEIGHBOURS_LINEWIDTH = 3
SNS_STYLE = {
    'grid.color': '#AAB3C8',
    'axes.facecolor': '#DCDBEE',
    'figure.facecolor' :'none',
}
DPI = 100


def plot_knn(ax, class_1, class_2, new_point, show_neighbours=True):
    if show_neighbours:
        neighbours = np.concatenate((class_1, [class_2[0], new_point]))
        neighbours_center = np.mean(neighbours, axis=0)
        neighbours_radius = np.amax(np.linalg.norm(neighbours - neighbours_center, axis=1)) * NEIGHBOURS_RADIUS_SCALE

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.scatter(
        class_1[:, 0],
        class_1[:, 1],
        color=CLASS_1_COLOR,
        marker=CLASS_1_MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
        label='Class 1',
    )
    ax.scatter(
        class_2[:, 0],
        class_2[:, 1],
        color=CLASS_2_COLOR,
        marker=CLASS_2_MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
        label='Class 2',
    )
    ax.scatter(
        new_point[0],
        new_point[1],
        color=NEW_POINT_COLOR,
        marker=NEW_POINT_MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
        label='New Point',
    )

    if show_neighbours:
        ax.add_patch(
            plt.Circle(
                neighbours_center,
                neighbours_radius,
                alpha=NEIGHBOURS_ALPHA,
                edgecolor=NEIGHBOURS_COLOR,
                facecolor='none',
                linewidth=NEIGHBOURS_LINEWIDTH,
            )
        )

    ax.legend(loc='upper left')


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    class_1 = np.array([(0.5, 1), (1, 1)])
    class_2 = np.array([(2, 1.5), (3, 3.5)])
    new_point = (1, 2)

    fig, ax = plt.subplots(1)
    plot_knn(ax, class_1, class_2, new_point, show_neighbours=False)
    plt.savefig(SCRIPT_PATH / 'knn.png', dpi=DPI)

    fig, ax = plt.subplots(1)
    plot_knn(ax, class_1, class_2, new_point, show_neighbours=True)
    plt.savefig(SCRIPT_PATH / 'knn_neighbours.png', dpi=DPI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
THEME = 'light'
XLIM = (4, 8)
YLIM = (1.75, 4.75)
XSTEP = 0.5
YSTEP = 0.25
XTICKS = np.arange(XLIM[0], XLIM[1] + XSTEP, XSTEP)
YTICKS = np.arange(YLIM[0], YLIM[1] + YSTEP, YSTEP)
CLASS_1_COLOR = '#4800FF' if THEME == 'light' else '#B699FF'
CLASS_2_COLOR = '#BF333C'
NEW_POINT_COLOR = '#00A058'
ALPHA = 1
NEIGHBOURS_COLOR = '#0A193B' if THEME == 'light' else '#A8BEF0'
NEIGHBOURS_ALPHA = 0.5
CLASS_1_MARKER = 'X'
CLASS_2_MARKER = 'o'
NEW_POINT_MARKER = '^'
MARKER_SIZE = 80
NEIGHBOURS_RADIUS_SCALE = 1.1
NEIGHBOURS_LINEWIDTH = 3
SNS_STYLE = {
    'grid.color': '#AAB3C8' if THEME == 'light' else '#474072',
    'axes.facecolor': '#DCDBEE' if THEME == 'light' else '#2C2847',
    'figure.facecolor' :'none',
    'xtick.color': '#000000' if THEME == 'light' else '#FFFFFF',
    'ytick.color': '#000000' if THEME == 'light' else '#FFFFFF',
}
LABEL_TEXT_COLOR = 'black' if THEME == 'light' else 'white'
LEGEND_TEXT_COLOR = 'black' if THEME == 'light' else 'white'
DPI = 100
SEED = 69


def load_iris_data():
    return pd.read_csv(SCRIPT_PATH / 'Iris.csv')


def plot_knn(
    ax,
    class_1,
    class_1_label,
    class_2,
    class_2_label,
    x_label,
    y_label,
    new_point=None,
    new_point_label=None,
    k=None,
    show_neighbours=True,
):
    if show_neighbours:
        all_classes = np.vstack((class_1, class_2))
        distances = np.linalg.norm(all_classes - new_point, axis=1)
        neighbours = all_classes[np.argsort(distances)[:k]]
        neighbours_center = np.mean(neighbours, axis=0)
        neighbours_radius = np.amax(np.linalg.norm(neighbours - neighbours_center, axis=1)) * NEIGHBOURS_RADIUS_SCALE

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.set_xlabel(x_label, color=LABEL_TEXT_COLOR)
    ax.set_ylabel(y_label, color=LABEL_TEXT_COLOR)

    ax.scatter(
        class_1[:, 0],
        class_1[:, 1],
        color=CLASS_1_COLOR,
        marker=CLASS_1_MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
        label=class_1_label,
    )
    ax.scatter(
        class_2[:, 0],
        class_2[:, 1],
        color=CLASS_2_COLOR,
        marker=CLASS_2_MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
        label=class_2_label,
    )
    if new_point is not None:
        ax.scatter(
            new_point[0],
            new_point[1],
            color=NEW_POINT_COLOR,
            marker=NEW_POINT_MARKER,
            s=MARKER_SIZE,
            alpha=ALPHA,
            label=new_point_label,
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

    ax.legend(labelcolor=LEGEND_TEXT_COLOR)


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    iris = load_iris_data()
    class_1_label = 'Iris-setosa'
    class_2_label = 'Iris-virginica'
    setosa = iris.loc[iris['Species'] == class_1_label]
    virginica = iris.loc[iris['Species'] == class_2_label]
    x_col = 'SepalLengthCm'
    y_col = 'SepalWidthCm'

    setosa_samples = setosa.sample(n=2, random_state=SEED)
    virginica_samples = virginica.sample(n=2, random_state=SEED)

    class_1 = setosa_samples[[x_col, y_col]].to_numpy(dtype=np.float64)
    class_2 = virginica_samples[[x_col, y_col]].to_numpy(dtype=np.float64)
    new_point = np.array([5.5, 3.2], dtype=np.float64)
    x_label = 'Sepal Length (cm)'
    y_label = 'Sepal Width (cm)'

    fig, ax = plt.subplots(1)
    plot_knn(
        ax,
        class_1=class_1,
        class_1_label=class_1_label,
        class_2=class_2,
        class_2_label=class_2_label,
        x_label=x_label,
        y_label=y_label,
        new_point=new_point,
        new_point_label='New Point',
        k=3,
        show_neighbours=False,
    )
    plt.savefig(SCRIPT_PATH / f'knn_{THEME}.png', dpi=DPI)

    fig, ax = plt.subplots(1)
    plot_knn(
        ax,
        class_1=class_1,
        class_1_label=class_1_label,
        class_2=class_2,
        class_2_label=class_2_label,
        x_label=x_label,
        y_label=y_label,
        new_point=new_point,
        new_point_label='New Point',
        k=3,
        show_neighbours=True,
    )
    plt.savefig(SCRIPT_PATH / f'knn_neighbours_{THEME}.png', dpi=DPI)

    class_1 = setosa[[x_col, y_col]].to_numpy(dtype=np.float64)
    class_2 = virginica[[x_col, y_col]].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(1)
    plot_knn(
        ax,
        class_1=class_1,
        class_1_label=class_1_label,
        class_2=class_2,
        class_2_label=class_2_label,
        x_label=x_label,
        y_label=y_label,
        new_point=new_point,
        new_point_label='New Point',
        k=8,
        show_neighbours=False,
    )
    plt.savefig(SCRIPT_PATH / f'knn_all_{THEME}.png', dpi=DPI)

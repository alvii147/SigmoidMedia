import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
NEUTRAL_COLOR = '#404040'
COLORS = ['#73009A', '#0A193B', '#FF4DA6']
CENTROID_COLOR = '#FF5C33'
CENTROID_EDGE_COLOR = '#4D0F00'
ALPHA = 0.8
CENTROID_ALPHA = 1
MARKER = 'o'
CENTROID_MARKER = 'X'
MARKER_SIZE = 80
CENTROID_MARKER_SIZE = 130
SNS_STYLE = {
    'grid.color': '#AAB3C8',
    'axes.facecolor': '#DCDBEE',
    'figure.facecolor' :'none',
}
DPI = 100


def load_iris_data():
    return pd.read_csv(SCRIPT_PATH / 'Iris.csv')


def plot_kmeans(
    ax,
    x,
    y,
    labels,
    x_label,
    y_label,
    centroids=None,
    legend=True,
):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        kwargs = {}
        colors = COLORS
        kwargs['label'] = f'Label {label}'
        if centroids is None:
            rng = np.random.default_rng(seed=42)
            p = rng.permutation(len(COLORS))
            colors = np.array(COLORS)[p]
            kwargs['label'] = label

        if not legend:
            colors = [NEUTRAL_COLOR] * len(COLORS)

        ax.scatter(
            x[idx],
            y[idx],
            c=colors[i],
            marker=MARKER,
            s=MARKER_SIZE,
            alpha=ALPHA,
            **kwargs
        )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            color=CENTROID_COLOR,
            edgecolor=CENTROID_EDGE_COLOR,
            marker=CENTROID_MARKER,
            s=CENTROID_MARKER_SIZE,
            linewidth=1,
            alpha=CENTROID_ALPHA,
            label='Centroid',
        )

    if legend:
        ax.legend()


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    iris = load_iris_data()
    x_col = 'SepalLengthCm'
    y_col = 'SepalWidthCm'
    label_col = 'Species'
    x_label = 'Sepal Length (cm)'
    y_label = 'Sepal Width (cm)'
    model = KMeans(n_clusters=3, random_state=42, n_init='auto')
    model.fit(iris[[x_col, y_col]])

    fig, ax = plt.subplots(1)
    plot_kmeans(
        ax=ax,
        x=iris[x_col],
        y=iris[y_col],
        labels=np.zeros(len(iris[x_col]), dtype=int),
        x_label=x_label,
        y_label=y_label,
        legend=False,
    )
    plt.savefig(SCRIPT_PATH / 'raw_data.png', dpi=DPI)

    fig, ax = plt.subplots(1)
    plot_kmeans(
        ax=ax,
        x=iris[x_col],
        y=iris[y_col],
        labels=model.labels_,
        x_label=x_label,
        y_label=y_label,
        centroids=model.cluster_centers_,
    )
    plt.savefig(SCRIPT_PATH / 'kmeans_predicted_labels.png', dpi=DPI)

    fig, ax = plt.subplots(1)
    plot_kmeans(
        ax=ax,
        x=iris[x_col],
        y=iris[y_col],
        labels=iris[label_col],
        x_label=x_label,
        y_label=y_label,
    )
    plt.savefig(SCRIPT_PATH / 'actual_labels.png', dpi=DPI)

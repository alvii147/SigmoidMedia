import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
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
    centroids,
    x_label,
    y_label,
):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.scatter(
        x,
        y,
        c=np.array(COLORS)[labels],
        marker=MARKER,
        s=MARKER_SIZE,
        alpha=ALPHA,
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color=CENTROID_COLOR,
        edgecolor=CENTROID_EDGE_COLOR,
        marker=CENTROID_MARKER,
        s=CENTROID_MARKER_SIZE,
        linewidth=1,
        alpha=CENTROID_ALPHA,
        label='Centroids',
    )
    ax.legend()


if __name__ == '__main__':
    sns.set_theme()
    sns.set_style('darkgrid', SNS_STYLE)

    iris = load_iris_data()
    x_col = 'SepalLengthCm'
    y_col = 'SepalWidthCm'
    x_label = 'Sepal Length (cm)'
    y_label = 'Sepal Width (cm)'
    model = KMeans(n_clusters=3, random_state=42, n_init='auto')
    model.fit(iris[[x_col, y_col]])

    fig, ax = plt.subplots(1)
    plot_kmeans(
        ax=ax,
        x=iris[x_col],
        y=iris[y_col],
        labels=model.labels_,
        centroids=model.cluster_centers_,
        x_label=x_label,
        y_label=y_label,
    )
    plt.savefig(SCRIPT_PATH / 'kmeans.png', dpi=DPI)

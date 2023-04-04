import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent
VALID_SUDOKU = np.array(
    [
        [7, 6, 8, 3, 5, 9, 2, 1, 4],
        [2, 1, 4, 7, 8, 6, 3, 9, 5],
        [3, 9, 5, 2, 4, 1, 7, 6, 8],
        [6, 5, 3, 9, 2, 4, 1, 8, 7],
        [9, 4, 2, 1, 7, 8, 6, 5, 3],
        [1, 8, 7, 6, 3, 5, 9, 4, 2],
        [4, 7, 1, 8, 6, 3, 5, 2, 9],
        [8, 3, 6, 5, 9, 2, 4, 7, 1],
        [5, 2, 9, 4, 1, 7, 8, 3, 6],
    ],
)
INVALID_SUDOKU = np.array(
    [
        [8, 6, 7, 2, 1, 5, 9, 4, 6],
        [9, 8, 4, 3, 7, 7, 2, 1, 5],
        [2, 5, 9, 9, 3, 8, 3, 7, 6],
        [6, 4, 3, 5, 2, 7, 8, 9, 1],
        [8, 1, 9, 6, 3, 4, 5, 2, 7],
        [5, 7, 2, 8, 5, 1, 5, 3, 4],
        [7, 9, 5, 1, 8, 2, 4, 6, 9],
        [1, 2, 8, 4, 6, 9, 7, 5, 3],
        [4, 9, 6, 7, 5, 3, 1, 8, 2],
    ],
)
DARK_TEXT_COLOR = '#0D001A'
LIGHT_TEXT_COLOR = '#F3E6FF'
DARK_BG_COLOR = '#C2ACD1'
LIGHT_BG_COLOR = '#9FC3E7'
HIGHLIGHT_BG_COLOR = '#0A193B'
MARK_BG_COLOR = '#B30059'
FONTSIZE = 12
FIGSIZE = (4, 4)
DPI = 100


def draw_sudoku(ax, sudoku, highlight_idx=None, highlight_type=None, mark_cells=[]):
    text_colors = np.full((9, 9), fill_value=DARK_TEXT_COLOR, dtype='U7')
    bg_colors = np.full((9, 9), fill_value=LIGHT_BG_COLOR, dtype='U7')
    for idx in range(0, 9, 2):
        i = (idx // 3) * 3
        j = (idx % 3) * 3
        bg_colors[i:i + 3, j:j + 3] = DARK_BG_COLOR

    if highlight_idx is not None and highlight_type is not None:
        if highlight_type.lower().strip() == 'row':
            text_colors[highlight_idx] = LIGHT_TEXT_COLOR
            bg_colors[highlight_idx] = HIGHLIGHT_BG_COLOR
        elif highlight_type.lower().strip() == 'column':
            text_colors[:, highlight_idx] = LIGHT_TEXT_COLOR
            bg_colors[:, highlight_idx] = HIGHLIGHT_BG_COLOR
        elif highlight_type.lower().strip() == 'square':
            i = (highlight_idx // 3) * 3
            j = (highlight_idx % 3) * 3
            text_colors[i:i + 3, j:j + 3] = LIGHT_TEXT_COLOR
            bg_colors[i:i + 3, j:j + 3] = HIGHLIGHT_BG_COLOR

    for cell_idx in mark_cells:
        text_colors[cell_idx] = LIGHT_TEXT_COLOR
        bg_colors[cell_idx] = MARK_BG_COLOR

    for i in range(9):
        for j in range(9):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_facecolor(bg_colors[i, j])
            ax[i, j].text(
                0.5,
                0.5,
                str(sudoku[i, j]),
                color=text_colors[i, j],
                fontsize=FONTSIZE,
                ha='center',
                va='center',
            )


if __name__ == '__main__':
    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, VALID_SUDOKU)
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'valid_sudoku.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, VALID_SUDOKU, highlight_idx=3, highlight_type='row')
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'valid_sudoku_row_highlighted.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, VALID_SUDOKU, highlight_idx=2, highlight_type='column')
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'valid_sudoku_column_highlighted.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, VALID_SUDOKU, highlight_idx=3, highlight_type='square')
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'valid_sudoku_square_highlighted.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, INVALID_SUDOKU)
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'invalid_sudoku.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, INVALID_SUDOKU, highlight_idx=6, highlight_type='row', mark_cells=[(6, 1), (6, 8)])
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'invalid_sudoku_row_highlighted.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, INVALID_SUDOKU, highlight_idx=5, highlight_type='column', mark_cells=[(1, 5), (3, 5)])
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'invalid_sudoku_column_highlighted.png', dpi=DPI)

    fig, ax = plt.subplots(9, 9, figsize=FIGSIZE, facecolor='none')
    draw_sudoku(ax, INVALID_SUDOKU, highlight_idx=5, highlight_type='square', mark_cells=[(4, 6), (5, 6)])
    plt.tight_layout()
    plt.savefig(SCRIPT_PATH / 'invalid_sudoku_square_highlighted.png', dpi=DPI)

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imgkit
import base64
import imageio.v2 as imageio


SCRIPT_PATH = Path(__file__).parent
IMG_PATH = SCRIPT_PATH / 'img'
KERNEL_PATH = SCRIPT_PATH / 'kernel'
MULTIPLICATION_PATH = SCRIPT_PATH / 'multiplication'
OUTPUT_PATH = SCRIPT_PATH / 'output'
HTML_PATH = SCRIPT_PATH / 'html'
FRAMES_PATH = SCRIPT_PATH / 'frames'

HTML = '''
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>Convolution</title>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <div style="display: -webkit-box; display: flex; -webkit-box-orient: horizontal; -webkit-box-direction: normal; flex-direction: row; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
            <div style="display: -webkit-box; display: flex; -webkit-box-orient: vertical; -webkit-box-direction: normal; flex-direction: column; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                <div style="display: -webkit-box; display: flex; -webkit-box-orient: horizontal; -webkit-box-direction: normal; flex-direction: row; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                    <div style="display: -webkit-box; display: flex; -webkit-box-orient: vertical; -webkit-box-direction: normal; flex-direction: column; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                        <h1 style="font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;">Kernel</h1>
                        <img src="data:image/png;base64,{b64kernel}" width="200px" />
                    </div>
                    <div style="display: -webkit-box; display: flex; -webkit-box-orient: vertical; -webkit-box-direction: normal; flex-direction: column; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                        <h1 style="font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;">Multiplication</h1>
                        <img src="data:image/png;base64,{b64multiplication}" width="200px" />
                    </div>
                </div>
                <div style="display: -webkit-box; display: flex; -webkit-box-orient: vertical; -webkit-box-direction: normal; flex-direction: column; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                    <h1 style="font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;">Image</h1>
                    <img src="data:image/png;base64,{b64img}" width="400px" />
                </div>
            </div>
            <div style="display: -webkit-box; display: flex; -webkit-box-orient: vertical; -webkit-box-direction: normal; flex-direction: column; -webkit-box-pack: center; justify-content: center; -webkit-box-align: center; align-items: center; padding: 20px;">
                <h1 style="font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;">Output</h1>
                <img src="data:image/png;base64,{b64output}" width="250px" />
            </div>
        </div>
    </body>
</html>
'''

IMG = np.array(
    [
        [-2, -1, 5, 1],
        [3, -2, 3, 4],
        [6, 4, -5, 4],
        [-3, 0, 0, 1],
    ],
    dtype=int,
)
KERNEL = np.array(
    [
        [1, 2],
        [0, -1],
    ],
    dtype=int,
)

DARK_COLOR1 = '#0D001A'
DARK_COLOR2 = '#0A193B'
DARK_COLOR3 = '#B30059'
DARK_COLOR4 = '#3D104B'
LIGHT_COLOR1 = '#F3E6FF'
LIGHT_COLOR2 = '#9FC3E7'
LIGHT_COLOR3 = '#C2ACD1'
FONT_SIZE_SMALL = 30
FONT_SIZE_LARGE = 52
FIGSIZE = (4, 4)
DPI = 100


def draw_grid(
    ax,
    values,
    text_color,
    bg_color,
    highlight_text_color,
    highlight_bg_color,
    font_size,
    highlight_idx=np.s_[[]],
):
    text_colors = np.full(values.shape, fill_value=text_color, dtype='U7')
    text_colors[highlight_idx] = highlight_text_color
    bg_colors = np.full(values.shape, fill_value=bg_color, dtype='U7')
    bg_colors[highlight_idx] = highlight_bg_color

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value is None:
                value = ''

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_facecolor(bg_colors[i, j])
            ax[i, j].text(
                0.5,
                0.5,
                value,
                color=text_colors[i, j],
                fontsize=font_size,
                weight='bold',
                ha='center',
                va='center',
            )


if __name__ == '__main__':
    frame = 0
    output = np.full(
        (
            IMG.shape[0] - KERNEL.shape[0] + 1,
            IMG.shape[1] - KERNEL.shape[1] + 1,
        ),
        fill_value=None,
        dtype=object,
    )

    for i in range(IMG.shape[0] - KERNEL.shape[0] + 1):
        for j in range(IMG.shape[1] - KERNEL.shape[1] + 1):
            frame += 1
            highlight_idx = np.s_[i:i + KERNEL.shape[0], j:j + KERNEL.shape[1]]
            multiplication = IMG[highlight_idx] * KERNEL
            output[i, j] = np.sum(multiplication)

            fig, ax = plt.subplots(*IMG.shape, figsize=FIGSIZE, facecolor='none')
            draw_grid(
                ax=ax,
                values=IMG,
                text_color=DARK_COLOR1,
                bg_color=LIGHT_COLOR2,
                highlight_text_color=LIGHT_COLOR1,
                highlight_bg_color=DARK_COLOR2,
                font_size=FONT_SIZE_SMALL,
                highlight_idx=highlight_idx,
            )
            plt.tight_layout()
            plt.savefig(IMG_PATH / f'img_{frame}.png', dpi=DPI)
            plt.close()

            fig, ax = plt.subplots(*multiplication.shape, figsize=FIGSIZE, facecolor='none')
            draw_grid(
                ax=ax,
                values=multiplication,
                text_color=DARK_COLOR1,
                bg_color=LIGHT_COLOR3,
                highlight_text_color=None,
                highlight_bg_color=None,
                font_size=FONT_SIZE_LARGE,
            )
            plt.tight_layout()
            plt.savefig(MULTIPLICATION_PATH / f'multiplication_{frame}.png', dpi=DPI)
            plt.close()

            fig, ax = plt.subplots(*output.shape, figsize=FIGSIZE, facecolor='none')
            draw_grid(
                ax=ax,
                values=output,
                text_color=LIGHT_COLOR1,
                bg_color=DARK_COLOR3,
                highlight_text_color=None,
                highlight_bg_color=None,
                font_size=FONT_SIZE_LARGE,
            )
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH / f'output_{frame}.png', dpi=DPI)
            plt.close()

    fig, ax = plt.subplots(*KERNEL.shape, figsize=FIGSIZE, facecolor='none')
    draw_grid(
        ax=ax,
        values=KERNEL,
        text_color=LIGHT_COLOR1,
        bg_color=DARK_COLOR4,
        highlight_text_color=None,
        highlight_bg_color=None,
        font_size=FONT_SIZE_LARGE,
    )
    plt.tight_layout()
    plt.savefig(KERNEL_PATH / f'kernel.png', dpi=DPI)
    plt.close()

    with open(KERNEL_PATH / 'kernel.png', 'rb') as f:
        b64kernel = base64.b64encode(f.read()).decode('utf-8')

    for i in range(1, frame + 1):
        with open(IMG_PATH / f'img_{i}.png', 'rb') as f:
            b64img = base64.b64encode(f.read()).decode('utf-8')

        with open(MULTIPLICATION_PATH / f'multiplication_{i}.png', 'rb') as f:
            b64multiplication = base64.b64encode(f.read()).decode('utf-8')

        with open(OUTPUT_PATH / f'output_{i}.png', 'rb') as f:
            b64output = base64.b64encode(f.read()).decode('utf-8')

        html_str = HTML.format(frame=i, b64kernel=b64kernel, b64img=b64img, b64multiplication=b64multiplication, b64output=b64output)
        with open(HTML_PATH / f'index_{i}.html', 'w') as f:
            f.write(html_str)

        try:
            imgkit.from_string(html_str, FRAMES_PATH / f'frame_{i}.png')
        except:
            pass

    imageio.mimsave(
        SCRIPT_PATH / 'animation.gif',
        [imageio.imread(FRAMES_PATH / frame_path) for frame_path in os.listdir(FRAMES_PATH)],
        format='GIF',
        loop=0,
        duration=2000,
    )

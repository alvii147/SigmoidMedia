import os
from pathlib import Path
import imageio.v2 as imageio

SCRIPT_PATH = Path(__file__).parent


if __name__ == '__main__':
    imageio.mimsave(
        SCRIPT_PATH / 'problem_skeleton.gif',
        [imageio.imread(SCRIPT_PATH / filepath) for filepath in os.listdir(SCRIPT_PATH) if filepath.endswith('.png')],
        format='GIF',
        loop=0,
        duration=500,
    )

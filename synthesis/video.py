"""Video export of the vocal tract animation.

Refactored from SynthsylShort/synthSYLVcourtes.py (playVLAMvid). The unused
GIF exporter playVLAM was removed (it also ignored its output path argument).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from synthesis.acoustic_model import freqevalNN
from synthesis.vlam import showgui


def playVLAMvid(output_video, fps, gui, Pval, valrect):
    """Render one VLAM sagittal frame per parameter row and write an AVI."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI
    video = None  # VideoWriter is initialized after the first frame

    for k in range(len(Pval)):
        _, _, _, gui, _ = freqevalNN(Pval[k, :], gui, valrect)
        fig = showgui(gui)

        # Convert the matplotlib figure to a numpy RGB image.
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]

        plt.close(fig)

        if video is None:
            height, width, _ = img.shape
            video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Convert RGB to BGR for OpenCV.
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

    if video is not None:
        video.release()

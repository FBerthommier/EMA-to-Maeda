"""VLAM video maker: animated midsagittal articulation video with audio.

Renders the VLAM midsagittal contour animation (matplotlib frames at
100 fps, 6x4 inch figures at 100 dpi, XVID AVI via OpenCV) for one
utterance and muxes the corresponding speech audio with ffmpeg to
produce a final MP4.

Two articulation sources are supported:

- ``copyparam`` (default): reference articulatory trajectories from
  ``data/copyparam/CP_2012.npy`` -- this reproduces the article video
  **CondA.mp4** (condition A).
- ``modelctwrandom``: converted CTW-random trajectories stored as a
  dictionary ``{'fval', 'Pv'}`` in ``MPFR_2012.npy`` (produced by
  ``vlam_audio_maker_v2.py``) -- this reproduces the article video
  **CondB.mp4** (condition B). ``mixAB.mp4`` is a side-by-side
  combination of the two.

Outputs:
- ``results/videos/vlm_output.avi`` (silent, 100 fps raw render)
- ``results/videos/vlm_output_dubbed.mp4`` (with audio from
  ``data/wav/vv_2012.wav``)

Required external tool: ffmpeg (on PATH, or set the FFMPEG constant in
``video_utils.py``).

Refactored from ``SynthsylShort/VLAMvidmaker2.py``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from video_utils import (
    DATA_DIR,
    FFMPEG,
    RESULTS_DIR,
    init_vlam_length,
    mux_audio_ffmpeg,
    render_video,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UTTERANCE = "2012"   # Stimulus name (utterance 2012 of the corpus).
FPS = 100            # Rendering frame rate (Hz).
VALRECT = 0.75       # Kept for API compatibility (unused by the forward model).
GUI_LEN = 195        # Vocal-tract length parameter for VLAM initialization.

PARAM_A_PATH = DATA_DIR / "copyparam" / f"CP_{UTTERANCE}.npy"          # Condition A.
PARAM_B_PATH = DATA_DIR / "modelctwrandom" / f"MPFR_{UTTERANCE}.npy"   # Condition B.
AUDIO_PATH = DATA_DIR / "wav" / f"vv_{UTTERANCE}.wav"                  # Speech audio.

AVI_PATH = RESULTS_DIR / "vlm_output.avi"
MP4_PATH = RESULTS_DIR / "vlm_output_dubbed.mp4"


def load_parameters(source):
    """Load articulatory trajectories (N, 7) for the requested condition."""
    if source == "copyparam":
        Pval = np.load(str(PARAM_A_PATH), allow_pickle=True)
    else:
        data = np.load(str(PARAM_B_PATH), allow_pickle=True).item()
        Pval = np.squeeze(data['Pv']).T
    return Pval


def main():
    parser = argparse.ArgumentParser(
        description="Render the VLAM midsagittal animation and mux the speech "
                    "audio (see CONFIG constants for defaults).")
    parser.add_argument("--source", choices=("copyparam", "modelctwrandom"),
                        default="copyparam",
                        help="Articulation source: copyparam = condition A "
                             "(CondA.mp4), modelctwrandom = condition B (CondB.mp4).")
    parser.add_argument("--avi", type=str, default=str(AVI_PATH),
                        help="Intermediate silent AVI output.")
    parser.add_argument("--mp4", type=str, default=str(MP4_PATH),
                        help="Final MP4 output with audio.")
    parser.add_argument("--ffmpeg", type=str, default=FFMPEG,
                        help="Path to the ffmpeg binary.")
    args = parser.parse_args()

    avi_path = Path(args.avi)
    mp4_path = Path(args.mp4)
    avi_path.parent.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    Pval = load_parameters(args.source)

    gui = init_vlam_length(np.zeros(7), GUI_LEN)

    print(f"Producing video at {FPS} Hz, frames={Pval.shape[0]} ...")
    render_video(avi_path, FPS, gui, Pval)
    print("Video produced.")

    # Mux the speech audio into the final MP4.
    print("Adding audio to video ...")
    try:
        mux_audio_ffmpeg(avi_path, AUDIO_PATH, mp4_path, ffmpeg_exe=args.ffmpeg)
        print(f"Dubbed video created: {mp4_path}")
    except Exception as e:
        print(f"Error while dubbing audio: {e}")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()

"""Convert aligned articulatory parameter tracks into formant tracks.

This is the shared conversion step of the condition A and condition B CTW
pipelines. A (7, T) aligned parameter track -- as stored in
``data/paramodelctw/PM_<name>.npy`` or
``data/paramodelctwrandom/PMR_<name>.npy`` -- is transposed to (T, 7),
cleaned of non-finite values, its Hyoid channel (column 6) is forced to
zero, and the VLAM articulatory model is evaluated frame by frame to
extract the F1-F3 formant trajectories. The formants and the converted
parameters are combined into the legacy dictionary format
``{'fval': (3, T, 1), 'Pv': (7, T)}`` of ``data/modelctw/MPF_<name>.npy``
and ``data/modelctwrandom/MPFR_<name>.npy``.

The computation is identical to ``video/vlam_audio_maker_v2.py``
(``process_one``: ``read_params_npy_7`` + ``syntform``); it is kept here
as an importable function so that the CTW pipeline scripts can write to
a configurable output root.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_DIR = REPO_ROOT / "video"
if str(_VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(_VIDEO_DIR))

from video_utils import init_vlam_length, syntform  # noqa: E402

# Soft-rectification constant and vocal-tract length for VLAM (same values
# as video/vlam_audio_maker_v2.py).
VALRECT = 0.75
GUI_LEN = 195


def convert_aligned_parameters_to_formants(param_track):
    """
    Convert one aligned parameter track into the MPF/MPFR dictionary.

    Applies exactly the rules of ``video_utils.read_params_npy_7``
    (transpose to (T, 7), clean non-finite values, force the Hyoid
    column to zero), then evaluates the VLAM model frame by frame.

    Args:
        param_track (np.ndarray): Aligned parameters of shape (7, T),
            e.g. the content of a PM_/PMR_ file or the array produced by
            the CTW alignment step.

    Returns:
        dict: ``{'fval': formants (3, T, 1), 'Pv': parameters (7, T)}``
        with the Hyoid row of ``Pv`` forced to zero, matching the
        archived ``data/modelctw`` / ``data/modelctwrandom`` files.
    """
    param_track = np.asarray(param_track, dtype=float)
    if param_track.ndim != 2 or param_track.shape[0] != 7:
        raise ValueError(f"Expected a (7, T) track, got {param_track.shape}")

    pval = np.nan_to_num(param_track.T, nan=0.0, posinf=0.0, neginf=0.0)
    pval[:, 6] = 0.0

    gui = init_vlam_length(np.zeros(7), GUI_LEN)
    fval = syntform(gui, pval, VALRECT)

    return {"fval": fval, "Pv": pval.T}

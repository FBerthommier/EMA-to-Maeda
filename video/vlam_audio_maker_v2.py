"""VLAM audio maker (version 2): formant computation for the CTW-random corpus.

For each stimulus ``{Y}0{X}`` (Y = speaker/condition index 2..10,
X = item index 10..16) this script:

1. Loads the 7-channel articulatory trajectories (transposed, Hyoid row
   zeroed) from ``data/paramodelctwrandom/PMR_<name>.npy``.
2. Runs the VLAM articulatory model frame by frame and computes the oral
   transfer functions and the F1-F3 formant trajectories
   (no waveform synthesis and no envelope shaping, unlike version 1).
3. Saves a combined dictionary ``{'fval': formants, 'Pv': parameters}``
   to ``results/videos/MPFR_<name>.npy`` (equivalent of the legacy
   ``modelctwrandom/MPFR_<name>.npy`` files).

These converted CTW-random trajectories are the condition B
articulation used in the article videos (``CondB.mp4`` and the
``mixAB.mp4`` side-by-side mix).

CONFIG constants below control input/output locations; edit them or
override with command-line flags (see ``--help``).

Refactored from ``SynthsylShort/VLAMaudmaker2.py``.
"""

import argparse
from pathlib import Path

import numpy as np

from video_utils import (
    DATA_DIR,
    RESULTS_DIR,
    init_vlam_length,
    read_params_npy_7,
    syntform,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VALRECT = 0.75  # Soft-rectification constant applied to area functions.
GUI_LEN = 195   # Vocal-tract length parameter for VLAM initialization.

X_RANGE = range(10, 17)  # Item index.
Y_RANGE = range(2, 11)   # Speaker/condition index.

PARAM_DIR = DATA_DIR / "paramodelctwrandom"  # PMR_<name>.npy trajectories.
OUT_DIR = RESULTS_DIR                        # Combined formant/parameter outputs.


def process_one(name, valrect, gui_len, out_dir):
    """Compute formants for one CTW-random stimulus and save the combined dict."""
    param_path = PARAM_DIR / f"PMR_{name}.npy"
    out_path = out_dir / f"MPFR_{name}.npy"

    # Load and prepare the (N, 7) parameter trajectories.
    Pval = read_params_npy_7(param_path)
    print(f"Loaded params: {Pval.shape[0]} frames x {Pval.shape[1]} params (should be 7)")

    Pval = np.nan_to_num(Pval, nan=0.0, posinf=0.0, neginf=0.0)

    # VLAM formant computation (7 parameters, Hyoid forced to zero).
    gui = init_vlam_length(np.zeros(7), gui_len)
    fval = syntform(gui, Pval, valrect)

    # Combined save of formants and (transposed) parameters.
    np.save(str(out_path), {'fval': fval, 'Pv': Pval.T})
    print(f"Saved {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="VLAM formant computation over the CTW-random corpus "
                    "(see CONFIG constants for defaults).")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help="Directory for the combined MPFR outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for X in X_RANGE:
        for Y in Y_RANGE:
            name = f"{Y}0{X}"
            process_one(name, VALRECT, GUI_LEN, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()

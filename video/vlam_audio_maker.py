"""VLAM audio maker (version 1): LPC articulatory synthesis for the CP corpus.

For each stimulus ``{Y}0{X}`` (Y = speaker/condition index 2..10,
X = item index 10..16) this script:

1. Loads the 6-channel articulatory trajectories in
   ``data/copyparam/CP_<name>.npy`` and appends a zero Hyoid channel.
2. Optionally resamples the trajectories from ``FS_IN`` to ``FS_OUT`` Hz.
3. Extracts the smoothed energy envelope (Hilbert, 8 Hz low-pass) of the
   corresponding real recording ``data/wav/vv_<name>.wav``.
4. Runs the VLAM articulatory model frame by frame, computes the oral
   transfer functions, and synthesizes speech with a time-varying LPC
   lattice excited by a glottal pulse train, shaped by the recorded
   envelope.
5. Saves the synthetic waveform to ``results/videos/CW_<name>.wav``
   (20 kHz) and the F1-F3 formant trajectories to
   ``results/videos/CF_<name>.npy``.

The resulting synthesized speech tracks are the audio used in the
article videos (``CondA.mp4``, ``CondB.mp4``, ``mixAB.mp4``).

CONFIG constants below control input/output locations and rates;
edit them or override with command-line flags (see ``--help``).

Refactored from ``SynthsylShort/VLAMaudmaker.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from video_utils import (
    DATA_DIR,
    RESULTS_DIR,
    extract_env,
    init_vlam_length,
    read_params_npy_6,
    resample_to_fs,
    syntwav,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FS_IN = 100.0      # Sampling rate of the input articulatory trajectories (Hz).
FS_OUT = 100.0     # Output parameter/synthesis frame rate (Hz).
VALRECT = 0.75     # Soft-rectification constant applied to area functions.
GUI_LEN = 195      # Vocal-tract length parameter for VLAM initialization.
SYNTH_FS = 20000   # Sampling rate of the synthesized waveforms (Hz).

X_RANGE = range(10, 17)  # Item index.
Y_RANGE = range(2, 11)   # Speaker/condition index.

PARAM_DIR = DATA_DIR / "copyparam"   # CP_<name>.npy articulatory trajectories.
WAV_DIR = DATA_DIR / "wav"           # vv_<name>.wav real recordings.
OUT_DIR = RESULTS_DIR                # Synthesized audio and formant outputs.


def process_one(name, fs_in, fs_out, valrect, gui_len, synth_fs, out_dir):
    """Synthesize audio and formants for one stimulus and save the results."""
    param_path = PARAM_DIR / f"CP_{name}.npy"
    wav_path = WAV_DIR / f"vv_{name}.wav"
    audio_out_path = out_dir / f"CW_{name}.wav"
    formant_out_path = out_dir / f"CF_{name}.npy"

    # Load and prepare the (N, 7) parameter trajectories.
    Pval = read_params_npy_6(param_path)
    print(f"Loaded params: {Pval.shape[0]} frames x {Pval.shape[1]} params (should be 7)")

    if fs_in != fs_out:
        Pval = resample_to_fs(Pval, fs_in=fs_in, fs_out=fs_out)
        print(f"Resampled to {Pval.shape[0]} frames ({fs_out} Hz)")
    else:
        print(f"No resampling needed (fs_in == fs_out == {fs_in} Hz)")

    Pval = np.nan_to_num(Pval, nan=0.0, posinf=0.0, neginf=0.0)

    # Energy envelope of the real recording (empirical 0.1 threshold).
    audio_data, fs = sf.read(str(wav_path))
    envsig = extract_env(audio_data, fs)
    envsig = np.maximum(0, envsig - 0.1)

    # VLAM synthesis.
    gui = init_vlam_length(np.zeros(7), gui_len)
    sig, fval = syntwav(gui, Pval, envsig, valrect)

    sf.write(str(audio_out_path), sig, synth_fs)
    np.save(str(formant_out_path), fval.T)
    print(f"Saved {audio_out_path.name} and {formant_out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="VLAM articulatory LPC synthesis over the CP corpus "
                    "(see CONFIG constants for defaults).")
    parser.add_argument("--fs-in", type=float, default=FS_IN,
                        help="Sampling rate of input parameters (Hz).")
    parser.add_argument("--fs-out", type=float, default=FS_OUT,
                        help="Output frame rate (Hz).")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help="Directory for synthesized audio and formants.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for X in X_RANGE:
        for Y in Y_RANGE:
            name = f"{Y}0{X}"
            process_one(name, args.fs_in, args.fs_out, VALRECT, GUI_LEN,
                        SYNTH_FS, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()

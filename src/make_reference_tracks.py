"""Regenerate the reference synthesis tracks (copywav / copyformants).

For every stimulus ``{Y}0{X}`` (X in 10..16, Y in 2..10) this script:

1. Loads the 6-channel reference parameters ``CP_<name>.npy`` (appending
   a zero Hyoid channel) from a configurable copyparam directory.
2. Extracts the smoothed energy envelope (Hilbert, 8 Hz low-pass,
   empirical 0.1 threshold) of the real recording
   ``data/wav/vv_<name>.wav``.
3. Runs the VLAM articulatory model frame by frame, computes the oral
   transfer functions and F1-F3 formants, and synthesizes speech with a
   time-varying LPC lattice excited by a glottal pulse train, shaped by
   the recorded envelope.
4. Saves the synthetic waveform to ``copywav/CW_<name>.wav`` (20 kHz,
   16-bit PCM -- the subtype of the archived files) and the formant
   trajectories to ``copyformants/CF_<name>.npy``.

This is the generator of the two archived reference-track sets; the
computation is the one implemented in ``video/vlam_audio_maker.py``
(process_one), itself refactored from the original
``SynthsylShort/VLAMaudmaker.py`` (lines 1255-1289). CF_ files are the
formants of the reference-driven VLAM synthesis -- i.e. the reference
side of the Figure 2b comparison is analyzed with the same VLAM
analyzer as the model side.

Outputs are written under a configurable root (default
``results/regenerated/data``) so the archived ``data/`` tree is never
overwritten.

Usage:
    python src/make_reference_tracks.py [--copyparam-dir DIR]
                                        [--out-root DIR]
                                        [--names 2012,3010,...] [--force]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_DIR = REPO_ROOT / "video"
if str(_VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(_VIDEO_DIR))

from video_utils import extract_env, init_vlam_length, read_params_npy_6, \
    syntwav  # noqa: E402

from ctw_tracks import stimulus_names  # noqa: E402

# Synthesis constants (same values as video/vlam_audio_maker.py).
FS_IN = 100.0      # Sampling rate of the input trajectories (Hz).
FS_OUT = 100.0     # Output parameter/synthesis frame rate (Hz).
VALRECT = 0.75     # Soft-rectification constant for the area functions.
GUI_LEN = 195      # Vocal-tract length parameter for VLAM.
SYNTH_FS = 20000   # Sampling rate of the synthesized waveforms (Hz).
WAV_SUBTYPE = "PCM_16"  # Subtype of the archived copywav files.

DEFAULT_COPYPARAM_DIR = REPO_ROOT / "data" / "copyparam"
DEFAULT_WAV_DIR = REPO_ROOT / "data" / "wav"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "regenerated" / "data"


def synthesize_track(name, copyparam_dir, wav_dir):
    """
    Synthesize the reference audio and formants of one stimulus.

    Mirrors ``video/vlam_audio_maker.py::process_one`` exactly (same
    constants, same operation order) but returns the results instead of
    writing them.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.
        wav_dir (Path): Directory holding the ``vv_<name>.wav``
            recordings.

    Returns:
        tuple:
            sig (np.ndarray): Synthetic waveform (float64).
            fval (np.ndarray): Formant trajectories of shape (3, T, 1).
    """
    param_path = Path(copyparam_dir) / f"CP_{name}.npy"
    wav_path = Path(wav_dir) / f"vv_{name}.wav"
    if not param_path.exists():
        raise FileNotFoundError(f"Missing input file: {param_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing input file: {wav_path}")

    pval = read_params_npy_6(param_path)
    if FS_IN != FS_OUT:
        from video_utils import resample_to_fs
        pval = resample_to_fs(pval, fs_in=FS_IN, fs_out=FS_OUT)
    pval = np.nan_to_num(pval, nan=0.0, posinf=0.0, neginf=0.0)

    # Energy envelope of the real recording (empirical 0.1 threshold).
    audio_data, fs = sf.read(str(wav_path))
    envsig = np.maximum(0, extract_env(audio_data, fs) - 0.1)

    gui = init_vlam_length(np.zeros(7), GUI_LEN)
    sig, fval = syntwav(gui, pval, envsig, VALRECT)
    return sig, fval


def save_track(name, sig, fval, out_root):
    """
    Save one stimulus pair into the canonical copywav/copyformants layout.

    Args:
        name (str): Stimulus name, e.g. "2012".
        sig (np.ndarray): Synthetic waveform.
        fval (np.ndarray): Formant trajectories of shape (3, T, 1).
        out_root (Path): Root receiving ``copywav/`` and
            ``copyformants/``.
    """
    wav_out = Path(out_root) / "copywav" / f"CW_{name}.wav"
    formant_out = Path(out_root) / "copyformants" / f"CF_{name}.npy"
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    formant_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_out), sig, SYNTH_FS, subtype=WAV_SUBTYPE)
    np.save(str(formant_out), fval.T)


def process_one(name, copyparam_dir, wav_dir, out_root, force=False):
    """
    Synthesize and save the CW_/CF_ files of one stimulus.

    Existing outputs are kept (the stimulus is skipped) unless ``force``
    is true.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.
        wav_dir (Path): Directory holding the ``vv_<name>.wav``
            recordings.
        out_root (Path): Root receiving ``copywav/`` and
            ``copyformants/``.
        force (bool): Recompute even if the outputs already exist.

    Returns:
        bool: True if the stimulus was computed, False if skipped.
    """
    wav_out = Path(out_root) / "copywav" / f"CW_{name}.wav"
    formant_out = Path(out_root) / "copyformants" / f"CF_{name}.npy"
    if not force and wav_out.exists() and formant_out.exists():
        print(f"[reference tracks] {name}: outputs already present, skipped")
        return False

    sig, fval = synthesize_track(name, copyparam_dir, wav_dir)
    save_track(name, sig, fval, out_root)
    print(f"[reference tracks] {name}: saved {wav_out.name} and "
          f"{formant_out.name}")
    return True


def main():
    """Run the reference-track generation over the requested stimuli."""
    parser = argparse.ArgumentParser(
        description="Reference-track generator: VLAM synthesis driven by the "
                    "reference parameters and the recorded envelope "
                    "(CW_/CF_ files).")
    parser.add_argument("--copyparam-dir", type=str,
                        default=str(DEFAULT_COPYPARAM_DIR),
                        help="Directory holding the CP_<name>.npy reference "
                             "parameters (default: data/copyparam).")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT),
                        help="Root for copywav/ and copyformants/ outputs "
                             "(default: results/regenerated/data).")
    parser.add_argument("--names", type=str, default=None,
                        help="Comma-separated subset of stimulus names "
                             "(default: all 63 stimuli).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute stimuli whose outputs already exist.")
    args = parser.parse_args()

    names = args.names.split(",") if args.names else stimulus_names()
    copyparam_dir = Path(args.copyparam_dir)
    if not copyparam_dir.is_dir():
        raise SystemExit(f"Missing input directory: {copyparam_dir}")

    for name in names:
        process_one(name, copyparam_dir, DEFAULT_WAV_DIR, Path(args.out_root),
                    force=args.force)

    print("Reference-track generation done.")


if __name__ == "__main__":
    main()

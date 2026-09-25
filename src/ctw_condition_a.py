"""Condition A pipeline: CTW alignment of the matching Maeda model.

For every stimulus ``{Y}0{X}`` (X in 10..16, Y in 2..10), the reference
Maeda model ``P0<X>`` is CTW-aligned onto the natural (EMA-derived)
parameters and the smoothed aligned track is saved as
``paramodelctw/PM_<name>.npy`` (shape (7, T)). The aligned track is then
converted through the VLAM model into F1-F3 formants and saved as the
legacy dictionary ``modelctw/MPF_<name>.npy`` (``{'fval', 'Pv'}``).

This is the generator of the two archived intermediate sets of condition
A; the alignment itself is the computation displayed (but not saved) by
``figures/figs1a_ctw_paths_condition_a.py``. Outputs are written under a
configurable root (default ``results/regenerated/data``) so the archived
``data/`` tree is never overwritten.

Usage:
    python src/ctw_condition_a.py [--copyparam-dir DIR] [--out-root DIR]
                                  [--names 2012,3010,...] [--force]
"""

import argparse
from pathlib import Path

import numpy as np

from ctw_tracks import align_model_to_natural, load_model_track, \
    load_natural_track, stimulus_names
from formant_conversion import convert_aligned_parameters_to_formants

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPYPARAM_DIR = REPO_ROOT / "data" / "copyparam"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "regenerated" / "data"


def condition_a_model_index(name):
    """
    Return the model index X of the condition A pairing for one stimulus.

    Condition A aligns each utterance Y0X onto its own reference model
    P0X. Names are built as ``str(Y) + "0" + str(X)`` with a two-digit X
    (10..16), so X is always the last two characters (this also holds
    for the two-digit speaker index Y = 10, e.g. "10016" -> X = 16).

    Args:
        name (str): Stimulus name, e.g. "2012" or "10016".

    Returns:
        int: The model index X (e.g. 12 or 16).
    """
    return int(name[-2:])


def generate_condition_a_track(name, copyparam_dir):
    """
    Compute the condition A aligned parameter track of one stimulus.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.

    Returns:
        np.ndarray: Smoothed aligned parameters of shape (7, T).
    """
    model_track = load_model_track(condition_a_model_index(name))
    natural_track = load_natural_track(name, copyparam_dir)
    return align_model_to_natural(model_track, natural_track)


def process_one(name, copyparam_dir, out_root, force=False):
    """
    Generate and save the PM_ and MPF_ files of one stimulus.

    Existing outputs are kept (the stimulus is skipped) unless ``force``
    is true.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.
        out_root (Path): Root receiving ``paramodelctw/`` and
            ``modelctw/``.
        force (bool): Recompute even if the outputs already exist.

    Returns:
        bool: True if the stimulus was computed, False if skipped.
    """
    pm_path = Path(out_root) / "paramodelctw" / f"PM_{name}.npy"
    mpf_path = Path(out_root) / "modelctw" / f"MPF_{name}.npy"
    if not force and pm_path.exists() and mpf_path.exists():
        print(f"[condition A] {name}: outputs already present, skipped")
        return False

    pm_track = generate_condition_a_track(name, copyparam_dir)
    mpf_dict = convert_aligned_parameters_to_formants(pm_track)

    pm_path.parent.mkdir(parents=True, exist_ok=True)
    mpf_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pm_path, pm_track)
    np.save(mpf_path, mpf_dict)
    print(f"[condition A] {name}: saved {pm_path.name} and {mpf_path.name}")
    return True


def main():
    """Run the condition A generation over the requested stimuli."""
    parser = argparse.ArgumentParser(
        description="Condition A generator: CTW-align model P0<X> onto each "
                    "utterance and convert the result to formants "
                    "(PM_/MPF_ files).")
    parser.add_argument("--copyparam-dir", type=str,
                        default=str(DEFAULT_COPYPARAM_DIR),
                        help="Directory holding the CP_<name>.npy natural "
                             "parameters (default: data/copyparam).")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT),
                        help="Root for paramodelctw/ and modelctw/ outputs "
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
        process_one(name, copyparam_dir, Path(args.out_root), force=args.force)

    print("Condition A generation done.")


if __name__ == "__main__":
    main()

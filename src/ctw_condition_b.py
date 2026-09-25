"""Condition B pipeline: CTW alignment of a random Maeda model.

For every stimulus ``{Y}0{X}`` (X in 10..16, Y in 2..10), a random model
``P0<X2>`` with X2 != X is CTW-aligned onto the natural (EMA-derived)
parameters and the smoothed aligned track is saved as
``paramodelctwrandom/PMR_<name>.npy`` (shape (7, T)). The aligned track
is then converted through the VLAM model into F1-F3 formants and saved as
the legacy dictionary ``modelctwrandom/MPFR_<name>.npy``
(``{'fval', 'Pv'}``).

The random pairing replicates exactly the seeded generator of
``figures/figs1b_ctw_paths_condition_b.py`` (``random.seed(126)``, one
draw per stimulus in the Y-outer/X-inner loop order), so the tracks
regenerated here are identical to those saved by that figure script.
Outputs are written under a configurable root (default
``results/regenerated/data``) so the archived ``data/`` tree is never
overwritten.

Usage:
    python src/ctw_condition_b.py [--copyparam-dir DIR] [--out-root DIR]
                                  [--names 2012,3010,...] [--force]
"""

import argparse
import random
from pathlib import Path

import numpy as np

from ctw_tracks import align_model_to_natural, load_model_track, \
    load_natural_track, stimulus_names
from formant_conversion import convert_aligned_parameters_to_formants

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COPYPARAM_DIR = REPO_ROOT / "data" / "copyparam"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "regenerated" / "data"

# Seed of the random model pairing (same as figures/figs1b_*.py).
RANDOM_SEED = 126


def pick_random_models(seed=RANDOM_SEED):
    """
    Build the stimulus -> random-model-index map of condition B.

    Replicates the generator of
    ``figures/figs1b_ctw_paths_condition_b.py`` call for call: seed 126,
    one ``random.choice`` per stimulus in the Y-outer/X-inner order.

    Args:
        seed (int): Seed of the legacy generator (default 126).

    Returns:
        dict: Maps each stimulus name to its random model index X2.
    """
    random.seed(seed)
    picks = {}
    for y in range(2, 11):
        for x in range(10, 17):
            name = f"{y}0{x}"
            picks[name] = random.choice([i for i in range(10, 17) if i != x])
    return picks


def generate_condition_b_track(name, copyparam_dir):
    """
    Compute the condition B aligned parameter track of one stimulus.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.

    Returns:
        np.ndarray: Smoothed aligned parameters of shape (7, T).
    """
    x2 = pick_random_models()[name]
    model_track = load_model_track(x2)
    natural_track = load_natural_track(name, copyparam_dir)
    return align_model_to_natural(model_track, natural_track)


def process_one(name, copyparam_dir, out_root, force=False):
    """
    Generate and save the PMR_ and MPFR_ files of one stimulus.

    Existing outputs are kept (the stimulus is skipped) unless ``force``
    is true.

    Args:
        name (str): Stimulus name, e.g. "2012".
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.
        out_root (Path): Root receiving ``paramodelctwrandom/`` and
            ``modelctwrandom/``.
        force (bool): Recompute even if the outputs already exist.

    Returns:
        bool: True if the stimulus was computed, False if skipped.
    """
    pmr_path = Path(out_root) / "paramodelctwrandom" / f"PMR_{name}.npy"
    mpfr_path = Path(out_root) / "modelctwrandom" / f"MPFR_{name}.npy"
    if not force and pmr_path.exists() and mpfr_path.exists():
        print(f"[condition B] {name}: outputs already present, skipped")
        return False

    pmr_track = generate_condition_b_track(name, copyparam_dir)
    mpfr_dict = convert_aligned_parameters_to_formants(pmr_track)

    pmr_path.parent.mkdir(parents=True, exist_ok=True)
    mpfr_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pmr_path, pmr_track)
    np.save(mpfr_path, mpfr_dict)
    print(f"[condition B] {name}: saved {pmr_path.name} and {mpfr_path.name}")
    return True


def main():
    """Run the condition B generation over the requested stimuli."""
    parser = argparse.ArgumentParser(
        description="Condition B generator: CTW-align a random model P0<X2> "
                    "onto each utterance (seed 126, as figs1b) and convert "
                    "the result to formants (PMR_/MPFR_ files).")
    parser.add_argument("--copyparam-dir", type=str,
                        default=str(DEFAULT_COPYPARAM_DIR),
                        help="Directory holding the CP_<name>.npy natural "
                             "parameters (default: data/copyparam).")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT),
                        help="Root for paramodelctwrandom/ and "
                             "modelctwrandom/ outputs (default: "
                             "results/regenerated/data).")
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

    print("Condition B generation done.")


if __name__ == "__main__":
    main()

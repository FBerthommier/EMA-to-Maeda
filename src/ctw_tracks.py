"""Shared loading and CTW-alignment helpers for the condition A/B tracks.

Both CTW pipeline scripts (``src/ctw_condition_a.py`` and
``src/ctw_condition_b.py``) align one Maeda reference model onto the
natural (EMA-derived) parameters of an utterance, smooth the aligned
track with a zero-phase low-pass filter, and store it transposed as a
(7, T) array. The loaders and the alignment step are factored out here so
that the two conditions share exactly the same numerical path (the
condition used by ``figures/figs1a_ctw_paths_condition_a.py`` and
``figures/figs1b_ctw_paths_condition_b.py``).
"""

from pathlib import Path

import numpy as np

from ctw import apply_ctw_path, zero_phase_lowpass_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
MAEDA_DIR = REPO_ROOT / "data" / "maeda"


def stimulus_names():
    """
    Build the list of the 63 stimulus names of the study.

    Returns:
        list of str: Names ``{Y}0{X}`` for X in 10..16 and Y in 2..10,
        in the (X outer, Y inner) order used by the original scripts.
    """
    return [f"{y}0{x}" for x in range(10, 17) for y in range(2, 11)]


def load_model_track(x_index):
    """
    Load the articulatory parameters of one Maeda reference model.

    Args:
        x_index (int): Model index, e.g. 12 for ``data/maeda/P012.npy``.

    Returns:
        np.ndarray: Parameters of shape (7, T1).
    """
    path = MAEDA_DIR / f"P0{x_index}.npy"
    data = np.load(path, allow_pickle=True).item()
    return np.squeeze(data["Pv"]).T


def load_natural_track(name, copyparam_dir):
    """
    Load the natural (EMA-derived) parameters of one utterance.

    Args:
        name (str): Utterance code, e.g. "2012" for Y=2, X=12.
        copyparam_dir (Path): Directory holding ``CP_<name>.npy``.

    Returns:
        np.ndarray: Parameters of shape (6, T2).
    """
    return np.load(Path(copyparam_dir) / f"CP_{name}.npy", allow_pickle=True).T


def align_model_to_natural(model_track, natural_track):
    """
    CTW-align a model track onto a natural track and smooth the result.

    The model (7, T1) is aligned onto the natural reference (6, T2); the
    aligned model is reconstructed on the T2 time grid, low-pass filtered
    (8 Hz zero-phase Butterworth, as in the figure scripts), and returned
    transposed as a (7, T2) array ready to be saved as PM_/PMR_.

    Args:
        model_track (np.ndarray): Model parameters of shape (7, T1).
        natural_track (np.ndarray): Natural parameters of shape (6, T2).

    Returns:
        np.ndarray: Smoothed aligned model parameters of shape (7, T2).
    """
    x_signal = model_track.T            # (T1, 7), as in figs1a/figs1b
    y_signal = natural_track[:6, :].T   # (T2, 6)
    x_aligned, _ = apply_ctw_path(x_signal, y_signal)
    x_aligned_smooth = zero_phase_lowpass_filter(x_aligned.T).T
    return x_aligned_smooth.T

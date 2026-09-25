"""Supplementary Figure S1a of the article: CTW warping paths for
condition A.

For every utterance (X in 10..16, Y in 2..11), the reference Maeda-model
parameters (P0X) are aligned onto the natural parameters (Y0X) and the
warping paths are superimposed in a single plot.

Original script: SynthsylShort/FigureS1a.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctw import apply_ctw_path
from plots import plot_ctw_path


def load_model_parameters(fname1):
    """
    Load the articulatory parameters of one Maeda reference model.

    Args:
        fname1 (str): Model code, e.g. "P012".

    Returns:
        np.ndarray: Parameters of shape (n_frames, n_params).
    """
    ema_data_path = REPO_ROOT / "data" / "maeda" / f"{fname1}.npy"
    data = np.load(ema_data_path, allow_pickle=True).item()
    return np.squeeze(data["Pv"]).T


def load_reference_parameters(fname2):
    """
    Load the reference (EMA-derived) articulatory parameters of one
    natural utterance.

    Args:
        fname2 (str): Utterance code, e.g. "2012" for Y=2, X=12.

    Returns:
        np.ndarray: Parameters of shape (6, n_frames).
    """
    ema_data_path = REPO_ROOT / "data" / "copyparam" / f"CP_{fname2}.npy"
    return np.load(str(ema_data_path), allow_pickle=True).T


def save_figure(fig, filename):
    """
    Save a figure to results/figures/ at 300 dpi (directory created if
    needed).

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filename (str): Output file name (PNG).
    """
    output_dir = REPO_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_dir / filename), dpi=300)
    print(f"Figure saved to {output_dir / filename}")


def main():
    """Build Figure S1a: all CTW paths for condition A."""
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for x in range(10, 17):  # X from 10 to 16 (outer loop)
        # Reference model name
        fname1 = f"P0{x}"
        ema_data1 = load_model_parameters(fname1)

        for y in range(2, 11):  # Y from 2 to 10 (inner loop)
            # Utterance name built with the Y0X rule
            fname2 = f"{y}0{x}"
            ema_data2 = load_reference_parameters(fname2)

            # Model signal (T1, 7) and reference signal (T2, 6)
            x_signal = ema_data1.T
            y_signal = ema_data2[:6, :].T

            # CTW alignment; plot the warping path
            _, path_matrix = apply_ctw_path(x_signal, y_signal)
            plot_ctw_path(path_matrix, color="lightgreen", ax=ax1)

    ax1.set_title("CTW paths for condition A")
    save_figure(fig1, "figs1a_ctw_paths_condition_a.png")
    plt.show()


if __name__ == "__main__":
    main()

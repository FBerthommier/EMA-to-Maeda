"""Supplementary Figure S1b of the article: CTW warping paths for
condition B.

For every utterance, the reference Maeda-model parameters of a random
model (X2 != X) are aligned onto the natural parameters and the warping
paths are superimposed in a single plot. The aligned (smoothed)
parameters are also saved to data/paramodelctwrandom/ and the
correlations between aligned and reference parameters are printed.
The random generator is seeded (126) for reproducibility.

Original script: SynthsylShort/FigureS1b.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from correlations import compute_correlation_simple
from ctw import apply_ctw_path, zero_phase_lowpass_filter
from plots import plot_ctw_path


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


def load_model_parameters(fname1):
    """
    Load the articulatory parameters of one Maeda reference model.

    Args:
        fname1 (str): Model code, e.g. "P015".

    Returns:
        np.ndarray: Parameters of shape (n_frames, n_params).
    """
    ema_data_path = REPO_ROOT / "data" / "maeda" / f"{fname1}.npy"
    data = np.load(ema_data_path, allow_pickle=True).item()
    return np.squeeze(data["Pv"]).T


def pick_random_model(x, rng):
    """
    Pick a random model index X2 in 10..16, different from x.

    Args:
        x (int): Model index to avoid.
        rng (random.Random): Random generator (seeded for
            reproducibility).

    Returns:
        int: The chosen model index X2.
    """
    possible_x2 = [i for i in range(10, 17) if i != x]
    return rng.choice(possible_x2)


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
    """Build Figure S1b and save the condition-B aligned parameters."""
    random.seed(126)  # For reproducibility
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    for y in range(2, 11):  # Y from 2 to 10 (outer loop)
        for x in range(10, 17):  # X from 10 to 16 (inner loop)
            # Utterance name built with the Y0X rule
            fname2 = f"{y}0{x}"
            ema_data2 = load_reference_parameters(fname2)

            # Random reference model, different from the natural model X
            x2 = pick_random_model(x, random)
            fname1 = f"P0{x2}"
            ema_data1 = load_model_parameters(fname1)

            # Output file for the aligned parameters
            copy_synth_path = (REPO_ROOT / "data" / "paramodelctwrandom" /
                               f"PMR_{fname2}.npy")

            # Model signal (T1, 7) and reference signal (T2, 6)
            x_signal = ema_data1.T
            y_signal = ema_data2[:6, :].T

            # CTW alignment, smoothing and save
            x_aligned, path_matrix = apply_ctw_path(x_signal, y_signal)
            x_aligned_smooth = zero_phase_lowpass_filter(x_aligned.T).T
            np.save(copy_synth_path, x_aligned_smooth.T)

            # Plot the warping path
            plot_ctw_path(path_matrix, color="lightblue", ax=ax1)

            # Correlation between the aligned and reference parameters
            corr_simple = compute_correlation_simple(
                x_aligned_smooth[:, :6],
                y_signal,
                labels=["J", "B", "D", "T", "LP", "LH"],
            )
            print(f"fname1={fname1}, fname2={fname2}, corr={corr_simple}")

    ax1.set_title("CTW paths for condition B")
    save_figure(fig1, "figs1b_ctw_paths_condition_b.png")
    plt.show()


if __name__ == "__main__":
    main()

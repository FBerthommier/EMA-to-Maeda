"""Supplementary Figure S3 of the article: comparison of articulatory
parameter trajectories (J, B, D, T, LP, LH) between the reference
synthesis (CS) and the CTW-aligned model, for one utterance (2012),
shown for condition A (CTW Model12) and condition B (CTW Model15).

Original script: SynthsylShort/FigureS3.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from plots import plot_articulatory_parameters

PARAM_LABELS = ["J", "B", "D", "T", "LP", "LH"]


def load_reference_parameters(fname):
    """
    Load the reference (EMA-derived) articulatory parameters of one
    natural utterance.

    Args:
        fname (str): Utterance code, e.g. "2012".

    Returns:
        np.ndarray: Parameters of shape (n_frames, 6).
    """
    copy_synth_path = REPO_ROOT / "data" / "copyparam" / f"CP_{fname}.npy"
    return np.load(str(copy_synth_path), allow_pickle=True)


def load_aligned_parameters(fname, condition):
    """
    Load the CTW-aligned model parameters for one utterance and
    condition.

    Args:
        fname (str): Utterance code, e.g. "2012".
        condition (str): "A" for modelctw (MPF files) or "B" for
            modelctwrandom (MPFR files).

    Returns:
        np.ndarray: Parameters of shape (n_frames, n_params).
    """
    if condition == "A":
        ctw_path = REPO_ROOT / "data" / "modelctw" / f"MPF_{fname}.npy"
    else:
        ctw_path = REPO_ROOT / "data" / "modelctwrandom" / f"MPFR_{fname}.npy"
    data = np.load(str(ctw_path), allow_pickle=True).item()
    return np.squeeze(data["Pv"]).T


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


def plot_condition(ema_data_ref, ema_data_aligned, fname, condition,
                   aligned_label):
    """
    Plot CS vs CTW-aligned parameter trajectories for one condition.

    Args:
        ema_data_ref (np.ndarray): Reference parameters, (T, 6).
        ema_data_aligned (np.ndarray): Aligned model parameters, (T2, N).
        fname (str): Utterance code used in the title.
        condition (str): Condition name used in the title ("A" or "B").
        aligned_label (str): Legend label of the aligned curves.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, axes = plot_articulatory_parameters(
        ema_data_ref[:, :6],
        labels=PARAM_LABELS,
        x_range=(0, ema_data_ref.shape[0]),
        y_range=(-4, 5),
        color="green" if condition == "A" else "blue",
        title=f"Comparison of Maeda parameters Cond {condition} (VV_{fname})",
        fig_number=1,
    )

    # Overlay the aligned trajectories
    plot_articulatory_parameters(
        ema_data_aligned[:, :6],
        new_figure=False,
        ax=axes,
        color="green" if condition == "A" else "blue",
        alpha=0.7,
        linewidth=0.8,
    )

    axes[0].legend(["CS", aligned_label], ncol=2, loc="upper left")
    return fig


def main():
    """Build Figure S3 for utterance 2012 (conditions A and B)."""
    fname = "2012"
    ema_data_ref = load_reference_parameters(fname)

    # --- Condition A: CTW with the natural model (Model12) ---
    ema_data_a = load_aligned_parameters(fname, "A")
    fig1 = plot_condition(ema_data_ref, ema_data_a, fname, "A", "CTW Model12")
    save_figure(fig1, "figs3_articulatory_param_trajectories_condA.png")
    plt.close(fig1)  # free figure number 1 before building condition B

    # --- Condition B: CTW with a random model (Model15) ---
    ema_data_b = load_aligned_parameters(fname, "B")
    fig2 = plot_condition(ema_data_ref, ema_data_b, fname, "B", "CTW Model15")
    save_figure(fig2, "figs3_articulatory_param_trajectories_condB.png")

    plt.show()


if __name__ == "__main__":
    main()

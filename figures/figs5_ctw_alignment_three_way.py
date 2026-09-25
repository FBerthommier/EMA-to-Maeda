"""Supplementary Figure S5 of the article: three-way comparison of the
CTW alignment of one Maeda reference model onto another (Model15
aligned onto Model12).

Shows the Model12 and Model15 parameters together with the CTW-aligned
Model15 parameters overlaid, and prints the Pearson correlations
between Model12/Model15 and the aligned signals.

Original script: SynthsylShort/FigureS5.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from correlations import compute_correlation_simple
from ctw import apply_ctw_path, zero_phase_lowpass_filter
from plots import plot_articulatory_parameters

PARAM_LABELS = ["J", "B", "D", "T", "LP", "LH"]


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
    ema_data = np.squeeze(data["Pv"])
    print(ema_data.shape)
    return ema_data


def align_and_smooth(x_signal, y_signal):
    """
    CTW-align x_signal onto y_signal and smooth the result.

    Args:
        x_signal (np.ndarray): Source signal, shape (T1, 6).
        y_signal (np.ndarray): Target signal, shape (T2, 6).

    Returns:
        np.ndarray: Aligned, smoothed signal of shape (T_tar, 6).
    """
    x_aligned, _ = apply_ctw_path(x_signal, y_signal)
    return zero_phase_lowpass_filter(x_aligned.T).T


def print_correlations(ema_data1, ema_data2, ema_data3, fname1, fname2):
    """
    Print the correlations before and after alignment.

    Args:
        ema_data1 (np.ndarray): Model12 parameters, (T1, N).
        ema_data2 (np.ndarray): Model15 parameters, (T2, N).
        ema_data3 (np.ndarray): Aligned Model15 parameters, (T3, N).
        fname1 (str): Code of the first model (target of alignment).
        fname2 (str): Code of the second model (source of alignment).
    """
    corr_simple = compute_correlation_simple(
        ema_data1[:, :6],
        ema_data2[:, :6],
        labels=PARAM_LABELS,
    )
    print(f"fname1={fname1}, fname2={fname2}, corr={corr_simple}")

    corr_simple = compute_correlation_simple(
        ema_data1[:, :6],
        ema_data3[:, :6],
        labels=PARAM_LABELS,
    )
    print(f"fname1={fname1}, Aligned, corr={corr_simple}")

    corr_simple = compute_correlation_simple(
        ema_data2[:, :6],
        ema_data3[:, :6],
        labels=PARAM_LABELS,
    )
    print(f"fname1={fname2}, Aligned, corr={corr_simple}")


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
    """Build Figure S5: Model12, Model15 and the CTW-aligned Model15."""
    fname1 = "P012"  # Model12 (alignment target)
    fname2 = "P015"  # Model15 (alignment source)

    ema_data1 = load_model_parameters(fname1)

    fig, axes = plot_articulatory_parameters(
        ema_data1[:, :6],
        labels=PARAM_LABELS,
        x_range=(0, ema_data1.shape[0]),
        y_range=(-4, 5),
        color="green",
        title="CTW-alignment of Model15 on Model12",
        fig_number=1,
    )

    ema_data2 = load_model_parameters(fname2)

    # Overlay the raw Model15 parameters
    plot_articulatory_parameters(
        ema_data2[:, :6],
        new_figure=False,
        ax=axes,
        color="blue",
        alpha=0.7,
        linewidth=0.8,
    )

    # CTW of Model15 (source) onto Model12 (target)
    x_signal = ema_data2[:, :6]
    y_signal = ema_data1[:, :6]
    ema_data3 = align_and_smooth(x_signal, y_signal)

    # Overlay the aligned parameters
    plot_articulatory_parameters(
        ema_data3[:, :6],
        new_figure=False,
        ax=axes,
        color="red",
        alpha=0.7,
        linewidth=0.8,
    )

    axes[0].legend(["Model12", "Model15", "CTW Model 15"],
                   ncol=3, loc="upper left")

    print_correlations(ema_data1, ema_data2, ema_data3, fname1, fname2)

    save_figure(fig, "figs5_ctw_alignment_three_way.png")
    plt.show()


if __name__ == "__main__":
    main()

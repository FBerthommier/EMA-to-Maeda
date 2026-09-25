"""Figure 3 of the article: example comparison of formant trajectories
(F1-F3, plotted from F3 on top down to F1 at the bottom) between the
reference synthesis (CS) and the CTW-aligned model synthesis (CTW) for
one utterance (X=12, Y=2).

Original script: SynthsylShort/Figure3.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 100  # Frame sampling rate in Hz


def load_formant_pair(fname2):
    """
    Load the reference (CS) and CTW-aligned model (CTW) formants for one
    utterance.

    Args:
        fname2 (str): Utterance code, e.g. "2012" for Y=2, X=12.

    Returns:
        tuple:
            f1 (np.ndarray): Reference formants, shape (T1, n_formants).
            f2 (np.ndarray): Model formants, shape (T2, n_formants).
    """
    file1 = REPO_ROOT / "data" / "copyformants" / f"CF_{fname2}.npy"
    data = np.load(str(file1), allow_pickle=True)
    f1 = np.squeeze(data)
    print(f"Shape: {f1.shape}")  # (n_frames, n_formants)

    file2 = REPO_ROOT / "data" / "modelctwrandom" / f"MPFR_{fname2}.npy"
    data = np.load(str(file2), allow_pickle=True).item()
    f2 = np.squeeze(data["fval"]).T
    print(f"Shape: {f2.shape}")  # (n_frames, n_formants)

    print(f"File 1: {f1.shape}, File 2: {f2.shape}")
    return f1, f2


def plot_formant_comparison(f1, f2, fname2):
    """
    Plot the formant trajectories of both sources (F3 on top, F1 on
    bottom).

    Args:
        f1 (np.ndarray): Reference formants, shape (T1, n_formants).
        f2 (np.ndarray): Model formants, shape (T2, n_formants).
        fname2 (str): Utterance code used in the title.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    n_formants = min(f1.shape[1], f2.shape[1])
    fig, axes = plt.subplots(n_formants, 1, figsize=(12, 3 * n_formants))

    if n_formants == 1:
        axes = [axes]

    # Time axis
    frames = min(f1.shape[0], f2.shape[0])
    time = np.arange(frames) / SAMPLE_RATE if SAMPLE_RATE else np.arange(frames)

    colors = plt.cm.tab10(np.linspace(0, 1, n_formants))

    # Inverted formant order: last formant on top, F1 at the bottom
    formant_indices = range(n_formants - 1, -1, -1)

    for ax_idx, formant_idx in enumerate(formant_indices):
        ax = axes[ax_idx]
        color = colors[formant_idx]

        ax.plot(time, f1[:frames, formant_idx], "-", linewidth=2,
                color=color, label="CS")
        ax.plot(time, f2[:frames, formant_idx], "--", linewidth=2,
                color=color, label="CTW")

        ax.set_ylabel(f"F{formant_idx + 1} (Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Time (s)" if SAMPLE_RATE else "Frames")
    plt.suptitle(f"Comparison of formants Cond A (VV_{fname2})", fontsize=18)
    plt.tight_layout()

    return fig


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
    """Build Figure 3 for utterance Y=2, X=12."""
    # Utterance selection: X = 12, Y = 2
    x = 12
    y = 2
    fname2 = f"{y}0{x}"

    f1, f2 = load_formant_pair(fname2)
    fig = plot_formant_comparison(f1, f2, fname2)

    save_figure(fig, "fig3_formant_trajectories.png")
    plt.show()


if __name__ == "__main__":
    main()

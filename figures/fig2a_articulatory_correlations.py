"""Figure 2a of the article: distribution of correlations between the
reference (EMA-derived) articulatory parameters and the CTW-aligned
Maeda-model parameters, for conditions A (green) and B (blue).

Pearson correlations are computed per utterance (X in 10..16, Y in 2..11)
for the six Maeda parameters, displayed as a violin plot, and compared
between conditions with paired t-tests corrected with the Holm procedure.

Original script: SynthsylShort/Figure2a.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from correlations import compute_correlation_simple
from stats_tests import paired_ttests_holm

PARAM_LABELS = ["J", "B", "D", "T", "LP", "LH"]
VIOLIN_LABELS = ["A/J", "B/J", "A/B", "B/B", "A/D", "B/D",
                 "A/T", "B/T", "A/LP", "B/LP", "A/LH", "B/LH"]
VIOLIN_POSITIONS = [0.5, 1, 2, 2.5, 3.5, 4, 5, 5.5, 6.5, 7, 8, 8.5]


def load_parameter_pair(fname2, condition):
    """
    Load the reference parameters and CTW-aligned model parameters for
    one utterance in one condition.

    Args:
        fname2 (str): Utterance code, e.g. "2012" for Y=2, X=12.
        condition (int): 0 for condition A (modelctw), 1 for condition B
            (modelctwrandom).

    Returns:
        tuple:
            params1 (np.ndarray): Reference parameters, shape (T1, 6).
            params2 (np.ndarray): CTW-aligned model parameters (first
                6 columns), shape (T2, 6).
    """
    file1 = REPO_ROOT / "data" / "copyparam" / f"CP_{fname2}.npy"
    data = np.load(str(file1), allow_pickle=True)
    params1 = np.squeeze(data)
    print(f"Shape: {params1.shape}")  # (n_frames, n_parameters)

    if condition == 0:
        file2 = REPO_ROOT / "data" / "modelctw" / f"MPF_{fname2}.npy"
    else:
        file2 = REPO_ROOT / "data" / "modelctwrandom" / f"MPFR_{fname2}.npy"

    data = np.load(str(file2), allow_pickle=True).item()
    params2 = np.squeeze(data["Pv"]).T
    params2 = params2[:, :6]
    print(f"Shape: {params2.shape}")  # (n_frames, n_parameters)

    return params1, params2


def compute_all_correlations():
    """
    Compute per-utterance correlations between reference and aligned
    model parameters for both conditions.

    Condition A fills the even columns of the correlation matrix and
    condition B the odd columns (interleaved per parameter).

    Returns:
        np.ndarray: Correlation matrix of shape (63, 12).
    """
    corr_matrix = np.zeros((63, 12))
    for condition in range(2):
        corr_accumulator = []

        for x in range(10, 17):  # X from 10 to 16 (outer loop)
            for y in range(2, 11):  # Y from 2 to 10 (inner loop)
                # Utterance name built with the Y0X rule
                fname2 = f"{y}0{x}"
                print(fname2)

                params1, params2 = load_parameter_pair(fname2, condition)
                corr_simple = compute_correlation_simple(
                    params1, params2, labels=PARAM_LABELS
                )
                corr_accumulator.append(corr_simple)

        # Interleave columns: A at 0+i, 2+i, ... and B at 1+i, 3+i, ...
        corr_matrix[:, [0 + condition, 2 + condition, 4 + condition,
                        6 + condition, 8 + condition, 10 + condition]] = \
            np.array(corr_accumulator)

    return corr_matrix


def print_summary_statistics(corr_matrix):
    """
    Print descriptive statistics for each condition/parameter column.

    Args:
        corr_matrix (np.ndarray): Correlation matrix of shape (63, 12).
    """
    print("\nStatistics per articulatory parameter:")
    print("=" * 60)
    for i, label in enumerate(VIOLIN_LABELS):
        col_data = corr_matrix[:, i]
        print(f"{label}:")
        print(f"  Mean = {np.mean(col_data):.4f}")
        print(f"  Variance = {np.var(col_data):.4f}")
        print(f"  Std = {np.std(col_data):.4f}")
        print(f"  Min/Max = {np.min(col_data):.4f}/{np.max(col_data):.4f}")
        print()

    print("\nComplete summary table:")
    print("Index\tParam\tMean\t\tVariance\tStd\t\tMin\t\tMax")
    for i, label in enumerate(VIOLIN_LABELS):
        col_data = corr_matrix[:, i]
        print(f"{i}\t{label}\t{np.mean(col_data):.4f}\t\t{np.var(col_data):.4f}"
              f"\t\t{np.std(col_data):.4f}\t\t{np.min(col_data):.4f}"
              f"\t{np.max(col_data):.4f}")


def plot_violin(corr_matrix):
    """
    Draw the violin plot of correlations by condition/parameter.

    Args:
        corr_matrix (np.ndarray): Correlation matrix of shape (63, 12).

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    data_violin = [corr_matrix[:, i] for i in range(12)]

    plt.figure(figsize=(8, 6))
    violin_parts = plt.violinplot(data_violin, positions=VIOLIN_POSITIONS,
                                  showmeans=False, showmedians=True)

    # Alternate colors: green for condition A, blue for condition B
    for i, pc in enumerate(violin_parts["bodies"]):
        if i % 2 == 0:
            pc.set_facecolor("lightgreen")
        else:
            pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)

    # Rotate labels so adjacent condition pairs do not collide
    plt.xticks(VIOLIN_POSITIONS, VIOLIN_LABELS, rotation=30, ha="right")
    plt.title("Distribution of correlations by condition/parameter")
    plt.xlabel("Condition/Parameter")
    plt.ylabel("Correlation coefficient")
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, 1.0)

    return plt.gcf()


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
    """Build Figure 2a and run the A vs B statistical comparison."""
    corr_matrix = compute_all_correlations()

    print_summary_statistics(corr_matrix)

    fig = plot_violin(corr_matrix)
    save_figure(fig, "fig2a_articulatory_correlations.png")
    plt.show()

    # Paired t-tests A vs B for each parameter, with Holm correction
    # Column pairs: (0,1)=J, (2,3)=B, (4,5)=D, (6,7)=T, (8,9)=LP, (10,11)=LH
    indices = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    paired_ttests_holm(corr_matrix, indices, PARAM_LABELS, row_label="Param")


if __name__ == "__main__":
    main()

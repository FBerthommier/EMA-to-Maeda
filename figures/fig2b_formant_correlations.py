"""Figure 2b of the article: distribution of correlations between the
reference formants (F1-F3) and the CTW-aligned model formants, for
conditions A (green) and B (blue).

Pearson correlations are computed per utterance (X in 10..16, Y in 2..11)
for F1-F3, displayed as a violin plot, and compared between conditions
with paired t-tests corrected with the Holm procedure.

Original script: SynthsylShort/Figure2b.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from correlations import compute_correlation_simple
from stats_tests import paired_ttests_holm

FORMANT_LABELS = ["F1", "F2", "F3"]
VIOLIN_LABELS = ["A/F1", "B/F1", "A/F2", "B/F2", "A/F3", "B/F3"]
VIOLIN_POSITIONS = [0.5, 1, 2, 2.5, 3.5, 4]


def load_formant_pair(fname2, condition):
    """
    Load the reference formants and CTW-aligned model formants for one
    utterance in one condition.

    Args:
        fname2 (str): Utterance code, e.g. "2012" for Y=2, X=12.
        condition (int): 0 for condition A (modelctw), 1 for condition B
            (modelctwrandom).

    Returns:
        tuple:
            formants1 (np.ndarray): Reference formants, shape (T1, 3).
            formants2 (np.ndarray): CTW-aligned model formants,
                shape (T2, 3).
    """
    file1 = REPO_ROOT / "data" / "copyformants" / f"CF_{fname2}.npy"
    data = np.load(str(file1), allow_pickle=True)
    formants1 = np.squeeze(data)
    print(f"Shape: {formants1.shape}")  # (n_frames, n_formants)

    if condition == 0:
        file2 = REPO_ROOT / "data" / "modelctw" / f"MPF_{fname2}.npy"
    else:
        file2 = REPO_ROOT / "data" / "modelctwrandom" / f"MPFR_{fname2}.npy"

    data = np.load(str(file2), allow_pickle=True).item()
    formants2 = np.squeeze(data["fval"]).T
    print(f"Shape: {formants2.shape}")  # (n_frames, n_formants)

    return formants1, formants2


def compute_all_correlations():
    """
    Compute per-utterance correlations between reference and aligned
    model formants for both conditions.

    Condition A fills the even columns of the correlation matrix and
    condition B the odd columns (interleaved per formant).

    Returns:
        np.ndarray: Correlation matrix of shape (63, 6).
    """
    corr_matrix = np.zeros((63, 6))
    for condition in range(2):
        corr_accumulator = []

        for x in range(10, 17):  # X from 10 to 16 (outer loop)
            for y in range(2, 11):  # Y from 2 to 10 (inner loop)
                # Utterance name built with the Y0X rule
                fname2 = f"{y}0{x}"
                print(fname2)

                formants1, formants2 = load_formant_pair(fname2, condition)
                corr_simple = compute_correlation_simple(
                    formants1, formants2, labels=FORMANT_LABELS, verbose=True
                )
                corr_accumulator.append(corr_simple)

        # Interleave columns: A at 0+i, 2+i, 4+i and B at 1+i, 3+i, 5+i
        corr_matrix[:, [0 + condition, 2 + condition, 4 + condition]] = \
            np.array(corr_accumulator)

    return corr_matrix


def print_summary_statistics(corr_matrix):
    """
    Print descriptive statistics for each condition/formant column.

    Args:
        corr_matrix (np.ndarray): Correlation matrix of shape (63, 6).
    """
    print("\nStatistics per formant:")
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
    print("Index\tFormant\tMean\t\tVariance\tStd\t\tMin\t\tMax")
    for i, label in enumerate(VIOLIN_LABELS):
        col_data = corr_matrix[:, i]
        print(f"{i}\t{label}\t{np.mean(col_data):.4f}\t\t{np.var(col_data):.4f}"
              f"\t\t{np.std(col_data):.4f}\t\t{np.min(col_data):.4f}"
              f"\t{np.max(col_data):.4f}")


def plot_violin(corr_matrix):
    """
    Draw the violin plot of correlations by condition/formant.

    Args:
        corr_matrix (np.ndarray): Correlation matrix of shape (63, 6).

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    data_violin = [corr_matrix[:, i] for i in range(6)]

    plt.figure(figsize=(6, 6))
    violin_parts = plt.violinplot(data_violin, positions=VIOLIN_POSITIONS,
                                  showmeans=False, showmedians=True)

    # Alternate colors: green for condition A, blue for condition B
    for i, pc in enumerate(violin_parts["bodies"]):
        if i % 2 == 0:
            pc.set_facecolor("lightgreen")
        else:
            pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)

    plt.xticks(VIOLIN_POSITIONS, VIOLIN_LABELS)
    plt.title("Distribution of correlations by condition/formant")
    plt.xlabel("Condition/Formant")
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
    """Build Figure 2b and run the A vs B statistical comparison."""
    corr_matrix = compute_all_correlations()

    print_summary_statistics(corr_matrix)

    fig = plot_violin(corr_matrix)
    save_figure(fig, "fig2b_formant_correlations.png")
    plt.show()

    # Paired t-tests A vs B for each formant, with Holm correction
    # Column pairs: (0,1)=F1, (2,3)=F2, (4,5)=F3
    indices = [(0, 1), (2, 3), (4, 5)]
    paired_ttests_holm(corr_matrix, indices, FORMANT_LABELS, row_label="Formant")


if __name__ == "__main__":
    main()

"""Pearson correlation helpers shared by the figure scripts."""

import numpy as np


def compute_correlation_simple(pv1, pv2, labels=None, verbose=False):
    """
    Compute per-column Pearson correlation coefficients between two matrices.

    The correlation is computed manually (mean/std/covariance) with
    sample standard deviations (ddof=1), matching the original research
    code. Columns with zero variance get a correlation of 0.

    Args:
        pv1 (np.ndarray): First matrix of shape (T, N).
        pv2 (np.ndarray): Second matrix of shape (T, N); must match pv1.
        labels (list of str or None): Optional names of the N columns,
            used only for display when verbose=True.
        verbose (bool): If True, print a table of correlations.

    Returns:
        np.ndarray: Vector of N correlation coefficients.
    """
    if pv1.shape != pv2.shape:
        raise ValueError(
            f"Matrices must have the same dimensions. pv1: {pv1.shape}, pv2: {pv2.shape}"
        )

    _, n_cols = pv1.shape

    if labels is None:
        labels = [f"Param_{i}" for i in range(n_cols)]

    correlations = np.zeros(n_cols)

    for i in range(n_cols):
        x = pv1[:, i]
        y = pv2[:, i]

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x, ddof=1)  # ddof=1 for the sample standard deviation
        y_std = np.std(y, ddof=1)

        # Avoid division by zero for constant columns
        if x_std == 0 or y_std == 0:
            correlations[i] = 0
        else:
            covariance = np.mean((x - x_mean) * (y - y_mean))
            correlations[i] = covariance / (x_std * y_std)

    if verbose:
        print("=" * 50)
        print("CORRELATION COEFFICIENTS PER PARAMETER")
        print("=" * 50)
        print(f"{'Parameter':<10} {'Correlation':<12}")
        print("-" * 50)
        for i in range(n_cols):
            print(f"{labels[i]:<10} {correlations[i]:<12.4f}")
        print("-" * 50)
        print(f"Mean  : {np.mean(correlations):.4f}")
        print(f"Median: {np.median(correlations):.4f}")
        print(f"Min   : {np.min(correlations):.4f} ({labels[np.argmin(correlations)]})")
        print(f"Max   : {np.max(correlations):.4f} ({labels[np.argmax(correlations)]})")
        print("=" * 50)

    return correlations

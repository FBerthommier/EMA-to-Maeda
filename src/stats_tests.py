"""Paired statistical tests comparing conditions A and B."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired_ttests_holm(corr_matrix, indices, param_names, row_label="Param"):
    """
    Run paired t-tests (condition A vs B) with Holm correction.

    For each parameter, a paired t-test is run between the condition A
    and condition B correlation columns, and p-values are corrected with
    the Holm procedure (family-wise error rate controlled at 0.05).
    Results are printed as a formatted table.

    Args:
        corr_matrix (np.ndarray): Matrix of shape (n_items, n_columns)
            with columns ordered as A/B pairs (e.g. 0:A/J, 1:B/J, ...).
        indices (list of tuple): (idx_A, idx_B) column pairs per parameter.
        param_names (list of str): Name of each parameter.
        row_label (str): Column header for the parameter names in the
            printed table (e.g. "Param" or "Formant").

    Returns:
        list of dict: One entry per parameter with keys
            "name", "mean_a", "mean_b", "mean_diff", "t_stat",
            "p_raw", "p_holm", "significant".
    """
    print(f"Comparison of conditions A vs B - paired t-test with Holm correction")
    print("=" * 90)
    print(f"{row_label:<6} {'Mean A':<8} {'Mean B':<8} {'Diff':<8} {'t-stat':<7} "
          f"{'p-raw':<9} {'p-Holm':<9} {'Sig (Holm)'}")
    print("-" * 90)

    # Raw paired t-tests
    results = []
    for (idx_a, idx_b), name in zip(indices, param_names):
        a = corr_matrix[:, idx_a]
        b = corr_matrix[:, idx_b]
        t_stat, p_value = stats.ttest_rel(a, b)
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        mean_diff = mean_a - mean_b
        results.append({
            "name": name,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "t_stat": t_stat,
            "p_raw": p_value,
        })

    # Holm correction (FWER controlled at 0.05)
    p_values = [r["p_raw"] for r in results]
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="holm")

    # Display with corrected p-values and significance
    for result, p_corr, is_sig in zip(results, p_corrected, reject):
        result["p_holm"] = p_corr
        result["significant"] = bool(is_sig)
        sig_text = "Yes" if is_sig else "No"
        print(f"{result['name']:<6} {result['mean_a']:<8.4f} {result['mean_b']:<8.4f} "
              f"{result['mean_diff']:<8.4f} {result['t_stat']:<7.3f} "
              f"{result['p_raw']:<9.4f} {p_corr:<9.4f} {sig_text}")

    # Summary of significant parameters after correction
    sig_params = [r["name"] for r in results if r["significant"]]
    print("\nParameters with a significant difference after Holm correction "
          "(corrected p < 0.05): "
          + (", ".join(sig_params) if sig_params else "None"))

    return results

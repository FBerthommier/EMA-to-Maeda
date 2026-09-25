"""Generate the reference statistics file for the reproducibility tests.

Runs the statistical analyses of Figures 2a and 2b (per-utterance
correlations between the reference and the CTW-aligned model, plus the
paired t-tests with Holm correction) directly from the archived
``data/`` tree and stores the numbers in
``tests/reference_stats.json``.

``tests/test_reproducibility.py`` re-runs the same computations and
compares them against this file, so any change in the numerical pipeline
is detected. Values are stored with 12 significant digits; the test
tolerance (1e-9) is far below that precision.

Usage (from the repository root):
    python tests/generate_reference_stats.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
OUTPUT_PATH = Path(__file__).resolve().parent / "reference_stats.json"

# t-test column pairs, duplicated here so this generator stays standalone.
FIG2A_INDICES = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
FIG2A_PARAMS = ["J", "B", "D", "T", "LP", "LH"]
FIG2B_INDICES = [(0, 1), (2, 3), (4, 5)]
FIG2B_PARAMS = ["F1", "F2", "F3"]


def load_figure_module(module_name):
    """
    Import one figure script as a module (they are not in a package).

    Args:
        module_name (str): Script file stem, e.g.
            "fig2a_articulatory_correlations".

    Returns:
        module: The imported module.
    """
    path = FIGURES_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def round12(value):
    """Round to 12 significant digits (keeps tiny p-values intact)."""
    return float(f"{float(value):.12g}")


def collect_analysis(module, indices, param_names, row_label):
    """
    Run one figure analysis and collect its numerical outputs.

    Args:
        module (module): Imported figure module (fig2a or fig2b).
        indices (list of tuple): (idx_A, idx_B) correlation column pairs.
        param_names (list of str): Parameter/formant names.
        row_label (str): Row header of the printed t-test table
            ("Param" or "Formant").

    Returns:
        dict: Violin labels, per-column means/stds, and the t-test table.
    """
    corr_matrix = module.compute_all_correlations()
    ttest_results = module.paired_ttests_holm(corr_matrix, indices,
                                              param_names, row_label=row_label)

    return {
        "violin_labels": module.VIOLIN_LABELS,
        "means": [round12(v) for v in np.mean(corr_matrix, axis=0)],
        "stds": [round12(v) for v in np.std(corr_matrix, axis=0)],
        "ttests": [
            {
                "name": r["name"],
                "mean_a": round12(r["mean_a"]),
                "mean_b": round12(r["mean_b"]),
                "mean_diff": round12(r["mean_diff"]),
                "t_stat": round12(r["t_stat"]),
                "p_raw": round12(r["p_raw"]),
                "p_holm": round12(r["p_holm"]),
                "significant": bool(r["significant"]),
            }
            for r in ttest_results
        ],
    }


def main():
    """Compute the Figure 2a/2b statistics and write reference_stats.json."""
    fig2a = load_figure_module("fig2a_articulatory_correlations")
    fig2b = load_figure_module("fig2b_formant_correlations")

    payload = {
        "description": "Reference statistics of Figures 2a/2b computed from "
                       "the archived data/ tree by "
                       "tests/generate_reference_stats.py",
        "fig2a": collect_analysis(fig2a, FIG2A_INDICES, FIG2A_PARAMS,
                                  "Param"),
        "fig2b": collect_analysis(fig2b, FIG2B_INDICES, FIG2B_PARAMS,
                                  "Formant"),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

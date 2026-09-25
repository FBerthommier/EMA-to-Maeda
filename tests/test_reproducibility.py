"""End-to-end reproducibility tests against the archived data.

Three families of checks (total runtime ~20 min, one core):

1. **Archive equality** — the intermediate-data generators
   (``src/make_reference_tracks.py``, ``src/ctw_condition_a.py``,
   ``src/ctw_condition_b.py``, ``src/batch_parameters.py``) are run on a
   subsample of stimuli and their outputs are compared with the archived
   ``data/`` files, at two levels:
   - *identity*: generators fed with the archived inputs (CP tracks)
     must reproduce the archives to machine precision;
   - *end-to-end*: generators fed with the CP tracks regenerated from
     the raw ``data/mat/`` corpus must reproduce the archives within
     documented tolerances (the raw-EMA reduction chain drifts by ~2e-5
     across NumPy/SciPy versions, and the CTW warping path is locally
     sensitive to such perturbations; see docs/PIPELINE.md).
2. **Statistics** — the Figure 2a/2b analyses (means, paired t-tests,
   Holm-corrected p-values) recomputed from ``data/`` must match
   ``tests/reference_stats.json`` to 1e-9.
3. **Figures** — every script of ``figures/`` must produce a non-empty
   PNG in ``results/figures/``.

Run from the repository root:
    python -m pytest tests/ -q
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from batch_parameters import load_ema_reduce, zero_phase_lowpass_filter
import ctw_condition_a
import ctw_condition_b
import make_reference_tracks

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
REFERENCE_STATS_PATH = Path(__file__).resolve().parent / "reference_stats.json"

# Subsample of stimuli used for the (expensive) regeneration checks,
# spread over the X and Y ranges of the corpus.
TEST_STIMULI = ["2012", "4010", "6013", "8015", "10016"]
# Stimuli additionally checked at the identity level (from archived CP).
IDENTITY_STIMULI = ["2012", "8015"]

# --- documented tolerances ------------------------------------------------
# Identity level (generators fed with the archived inputs): machine
# precision, except the waveforms (16-bit PCM quantization).
# Measured maxima on this repository: 3.5e-14 (parameters), 2.3e-11 Hz
# (formants), 1 LSB (waveforms).
CF_IDENTITY_ATOL = 1e-9        # Hz, formant tracks
PM_IDENTITY_ATOL = 1e-9        # parameter units, aligned tracks
FVAL_IDENTITY_ATOL = 1e-9      # Hz, MPF/MPFR formants
CW_IDENTITY_MAX_LSB = 2        # least-significant bits of the PCM_16 wav

# End-to-end level (from the raw EMA corpus), measured on the 5 test
# stimuli: the CP regeneration drifts by <= 1.8e-5 across library
# versions (float32 EMA reduction + decimation); the reference formants
# stay within 3.1e-3 Hz and the waveforms within 1 LSB. The CTW warping
# path can flip in near-tie regions under such perturbations: at most
# 2.3% of the aligned frames deviate by more than 0.05 parameter units
# and at most 2.8% by more than 10 Hz (maxima ~0.3 and ~100 Hz on those
# few frames). Tolerances below keep a margin over these measurements.
CP_E2E_ATOL = 5e-5             # parameter units
CF_E2E_ATOL = 0.1              # Hz
CW_E2E_MAX_LSB = 4             # least-significant bits (~1.2e-4 amplitude)
PM_E2E_ATOL = 0.5              # parameter units, per-frame maximum
PM_E2E_FRAME_ATOL = 0.05       # parameter units, affected-frame threshold
PM_E2E_FRAME_FRACTION = 0.05   # max fraction of frames above PM_E2E_FRAME_ATOL
FVAL_E2E_ATOL = 150.0          # Hz, per-frame maximum
FVAL_E2E_FRAME_ATOL = 10.0     # Hz, affected-frame threshold
FVAL_E2E_FRAME_FRACTION = 0.05  # max fraction of frames above 10 Hz


def regenerate_copyparam(name):
    """
    Regenerate one CP track in memory, straight from the raw EMA .mat.

    Applies the exact pipeline of ``src/batch_parameters.py``
    (``process_utterance``) without writing anything.
    """
    from scipy import signal

    model = np.load(DATA_DIR / "maeda" / f"P0{name[-2:]}.npy",
                    allow_pickle=True).item()
    ema = np.squeeze(model["Pv"]).T
    mean1 = np.mean(ema, axis=1, keepdims=True)[:6]
    std1 = np.std(ema, axis=1, keepdims=True)[:6]

    brut = load_ema_reduce(DATA_DIR / "mat" / f"VV_{name}.mat")
    down = signal.decimate(brut, 5, axis=1, n=4)
    filtered = zero_phase_lowpass_filter(down)
    z = (filtered - np.mean(filtered, axis=1, keepdims=True)) / \
        (np.std(filtered, axis=1, keepdims=True) + 1e-9)
    return ((z * std1) + mean1).T


def quantize_pcm16(sig):
    """Quantize a float waveform exactly as a PCM_16 wav write does."""
    q = (np.clip(sig, -1.0, 1.0) * 32768).round()
    return np.clip(q, -32768, 32767).astype(np.int16)


def max_abs_diff(regenerated, archived):
    """Maximum absolute difference between two arrays."""
    return float(np.max(np.abs(np.asarray(regenerated, dtype=float)
                               - np.asarray(archived, dtype=float))))


def assert_track_within_tolerance(regenerated, archived, max_atol,
                                  frame_atol, frame_fraction, label):
    """
    Compare one aligned track (parameters or formants) with its archive.

    Checks both the per-frame maximum deviation and the fraction of
    frames deviating beyond ``frame_atol`` (the CTW path flips only a
    few frames when the CP input drifts by ~1e-5; see the tolerance
    block above).

    Args:
        regenerated (np.ndarray): Regenerated track, shape (D, T).
        archived (np.ndarray): Archived track, same shape.
        max_atol (float): Tolerance on the per-frame maximum.
        frame_atol (float): Per-frame deviation threshold.
        frame_fraction (float): Max fraction of frames above frame_atol.
        label (str): Identification in the failure message.
    """
    diff = np.abs(np.asarray(regenerated, dtype=float)
                  - np.asarray(archived, dtype=float))
    per_frame = diff.reshape(diff.shape[0], -1).max(axis=0)
    assert per_frame.max() <= max_atol, \
        f"{label}: max per-frame deviation {per_frame.max():.3e} > {max_atol}"
    fraction = float(np.mean(per_frame > frame_atol))
    assert fraction <= frame_fraction, \
        f"{label}: {fraction * 100:.1f}% of frames deviate by more than " \
        f"{frame_atol} (limit {frame_fraction * 100:.0f}%)"


@pytest.fixture(scope="session")
def regenerated():
    """
    Run the full regeneration chain once for the test stimuli.

    Returns a dict holding, per stimulus, the regenerated arrays (from
    the raw corpus) and, for IDENTITY_STIMULI, the identity-level arrays
    (generators fed with the archived CP tracks).
    """
    out = {}
    for name in TEST_STIMULI:
        entry = {}
        cp_reg = regenerate_copyparam(name)
        entry["cp"] = cp_reg

        # Reference tracks from the regenerated CP (the loader is patched
        # to return the in-memory array instead of reading the file).
        original_loader = make_reference_tracks.read_params_npy_6
        make_reference_tracks.read_params_npy_6 = \
            lambda _p, _cp=cp_reg: np.hstack([_cp, np.zeros((_cp.shape[0], 1))])
        try:
            sig, fval = make_reference_tracks.synthesize_track(
                name, DATA_DIR / "copyparam", DATA_DIR / "wav")
        finally:
            make_reference_tracks.read_params_npy_6 = original_loader
        entry["cw_sig"] = sig
        entry["cf"] = fval.T

        # Condition A from the regenerated CP.
        original_loader = ctw_condition_a.load_natural_track
        ctw_condition_a.load_natural_track = \
            lambda _n, _d, _cp=cp_reg: _cp.T
        try:
            pm = ctw_condition_a.generate_condition_a_track(
                name, DATA_DIR / "copyparam")
        finally:
            ctw_condition_a.load_natural_track = original_loader
        entry["pm"] = pm
        entry["mpf"] = ctw_condition_a.convert_aligned_parameters_to_formants(pm)

        # Condition B from the regenerated CP.
        original_loader = ctw_condition_b.load_natural_track
        ctw_condition_b.load_natural_track = \
            lambda _n, _d, _cp=cp_reg: _cp.T
        try:
            pmr = ctw_condition_b.generate_condition_b_track(
                name, DATA_DIR / "copyparam")
        finally:
            ctw_condition_b.load_natural_track = original_loader
        entry["pmr"] = pmr
        entry["mpfr"] = ctw_condition_b.convert_aligned_parameters_to_formants(pmr)

        if name in IDENTITY_STIMULI:
            sig_i, fval_i = make_reference_tracks.synthesize_track(
                name, DATA_DIR / "copyparam", DATA_DIR / "wav")
            pm_i = ctw_condition_a.generate_condition_a_track(
                name, DATA_DIR / "copyparam")
            pmr_i = ctw_condition_b.generate_condition_b_track(
                name, DATA_DIR / "copyparam")
            entry["identity"] = {
                "cw_sig": sig_i,
                "cf": fval_i.T,
                "pm": pm_i,
                "mpf": ctw_condition_a.convert_aligned_parameters_to_formants(pm_i),
                "pmr": pmr_i,
                "mpfr": ctw_condition_b.convert_aligned_parameters_to_formants(pmr_i),
            }

        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# 1. Archive equality — copyparam (from the raw EMA corpus)
# ---------------------------------------------------------------------------
def test_copyparam_end_to_end(regenerated):
    """CP regenerated from data/mat matches the archive within 5e-5."""
    for name in TEST_STIMULI:
        archived = np.load(DATA_DIR / "copyparam" / f"CP_{name}.npy")
        diff = max_abs_diff(regenerated[name]["cp"], archived)
        assert diff <= CP_E2E_ATOL, f"CP_{name}: max|diff| = {diff:.3e}"


# ---------------------------------------------------------------------------
# 1. Archive equality — reference tracks (CW/CF)
# ---------------------------------------------------------------------------
def test_reference_tracks_identity(regenerated):
    """From the archived CP, CW/CF reproduce the archive exactly."""
    for name in IDENTITY_STIMULI:
        ident = regenerated[name]["identity"]
        cf_archived = np.load(DATA_DIR / "copyformants" / f"CF_{name}.npy")
        diff = max_abs_diff(ident["cf"], cf_archived)
        assert diff <= CF_IDENTITY_ATOL, f"CF_{name}: max|diff| = {diff:.3e}"

        wav_archived, _ = sf.read(str(DATA_DIR / "copywav" / f"CW_{name}.wav"),
                                  dtype="int16")
        lsb = int(np.max(np.abs(quantize_pcm16(ident["cw_sig"]).astype(int)
                                - wav_archived.astype(int))))
        assert lsb <= CW_IDENTITY_MAX_LSB, f"CW_{name}: {lsb} LSB off"


def test_reference_tracks_end_to_end(regenerated):
    """From the raw corpus, CW/CF match the archive within tolerance."""
    for name in TEST_STIMULI:
        entry = regenerated[name]
        cf_archived = np.load(DATA_DIR / "copyformants" / f"CF_{name}.npy")
        diff = max_abs_diff(entry["cf"], cf_archived)
        assert diff <= CF_E2E_ATOL, f"CF_{name}: max|diff| = {diff:.3e} Hz"

        wav_archived, _ = sf.read(str(DATA_DIR / "copywav" / f"CW_{name}.wav"),
                                  dtype="int16")
        lsb = int(np.max(np.abs(quantize_pcm16(entry["cw_sig"]).astype(int)
                                - wav_archived.astype(int))))
        assert lsb <= CW_E2E_MAX_LSB, f"CW_{name}: {lsb} LSB off"


# ---------------------------------------------------------------------------
# 1. Archive equality — CTW condition A (PM/MPF)
# ---------------------------------------------------------------------------
def test_ctw_condition_a_identity(regenerated):
    """From the archived CP, PM/MPF reproduce the archive exactly."""
    for name in IDENTITY_STIMULI:
        ident = regenerated[name]["identity"]
        pm_archived = np.load(DATA_DIR / "paramodelctw" / f"PM_{name}.npy")
        mpf_archived = np.load(DATA_DIR / "modelctw" / f"MPF_{name}.npy",
                               allow_pickle=True).item()
        assert max_abs_diff(ident["pm"], pm_archived) <= PM_IDENTITY_ATOL
        assert max_abs_diff(ident["mpf"]["Pv"], mpf_archived["Pv"]) \
            <= PM_IDENTITY_ATOL
        assert max_abs_diff(ident["mpf"]["fval"], mpf_archived["fval"]) \
            <= FVAL_IDENTITY_ATOL


def test_ctw_condition_a_end_to_end(regenerated):
    """From the raw corpus, PM/MPF match the archive within tolerance."""
    for name in TEST_STIMULI:
        entry = regenerated[name]
        pm_archived = np.load(DATA_DIR / "paramodelctw" / f"PM_{name}.npy")
        mpf_archived = np.load(DATA_DIR / "modelctw" / f"MPF_{name}.npy",
                               allow_pickle=True).item()
        assert_track_within_tolerance(
            entry["pm"], pm_archived, PM_E2E_ATOL, PM_E2E_FRAME_ATOL,
            PM_E2E_FRAME_FRACTION, f"PM_{name}")
        assert_track_within_tolerance(
            entry["mpf"]["Pv"], mpf_archived["Pv"], PM_E2E_ATOL,
            PM_E2E_FRAME_ATOL, PM_E2E_FRAME_FRACTION, f"MPF_{name} Pv")
        assert_track_within_tolerance(
            entry["mpf"]["fval"], mpf_archived["fval"], FVAL_E2E_ATOL,
            FVAL_E2E_FRAME_ATOL, FVAL_E2E_FRAME_FRACTION, f"MPF_{name} fval")


# ---------------------------------------------------------------------------
# 1. Archive equality — CTW condition B (PMR/MPFR)
# ---------------------------------------------------------------------------
def test_condition_b_pairing():
    """The seed-126 random pairing is deterministic and matches figs1b."""
    picks = ctw_condition_b.pick_random_models()
    assert picks["2012"] == 15  # cross-checked with supplement Fig. S3
    assert ctw_condition_b.pick_random_models() == picks


def test_ctw_condition_b_identity(regenerated):
    """From the archived CP, PMR/MPFR reproduce the archive exactly."""
    for name in IDENTITY_STIMULI:
        ident = regenerated[name]["identity"]
        pmr_archived = np.load(DATA_DIR / "paramodelctwrandom" /
                               f"PMR_{name}.npy")
        mpfr_archived = np.load(DATA_DIR / "modelctwrandom" /
                                f"MPFR_{name}.npy", allow_pickle=True).item()
        assert max_abs_diff(ident["pmr"], pmr_archived) <= PM_IDENTITY_ATOL
        assert max_abs_diff(ident["mpfr"]["Pv"], mpfr_archived["Pv"]) \
            <= PM_IDENTITY_ATOL
        assert max_abs_diff(ident["mpfr"]["fval"], mpfr_archived["fval"]) \
            <= FVAL_IDENTITY_ATOL


def test_ctw_condition_b_end_to_end(regenerated):
    """From the raw corpus, PMR/MPFR match the archive within tolerance."""
    for name in TEST_STIMULI:
        entry = regenerated[name]
        pmr_archived = np.load(DATA_DIR / "paramodelctwrandom" /
                               f"PMR_{name}.npy")
        mpfr_archived = np.load(DATA_DIR / "modelctwrandom" /
                                f"MPFR_{name}.npy", allow_pickle=True).item()
        assert_track_within_tolerance(
            entry["pmr"], pmr_archived, PM_E2E_ATOL, PM_E2E_FRAME_ATOL,
            PM_E2E_FRAME_FRACTION, f"PMR_{name}")
        assert_track_within_tolerance(
            entry["mpfr"]["Pv"], mpfr_archived["Pv"], PM_E2E_ATOL,
            PM_E2E_FRAME_ATOL, PM_E2E_FRAME_FRACTION, f"MPFR_{name} Pv")
        assert_track_within_tolerance(
            entry["mpfr"]["fval"], mpfr_archived["fval"], FVAL_E2E_ATOL,
            FVAL_E2E_FRAME_ATOL, FVAL_E2E_FRAME_FRACTION,
            f"MPFR_{name} fval")


# ---------------------------------------------------------------------------
# 2. Statistics of Figures 2a/2b
# ---------------------------------------------------------------------------
def load_reference_stats():
    with open(REFERENCE_STATS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_figure_module(module_name):
    """Import one figure script as a module (they are not a package)."""
    path = FIGURES_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def figure_statistics():
    """Compute the Figure 2a/2b analyses once from the archived data."""
    stats = {}
    for module_name, indices, params, row_label in (
            ("fig2a_articulatory_correlations",
             [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
             ["J", "B", "D", "T", "LP", "LH"], "Param"),
            ("fig2b_formant_correlations",
             [(0, 1), (2, 3), (4, 5)], ["F1", "F2", "F3"], "Formant")):
        module = load_figure_module(module_name)
        corr = module.compute_all_correlations()
        ttests = module.paired_ttests_holm(corr, indices, params,
                                           row_label=row_label)
        stats[module_name[:5]] = {
            "means": np.mean(corr, axis=0),
            "stds": np.std(corr, axis=0),
            "ttests": ttests,
        }
    return stats


STATS_ATOL = 1e-9


@pytest.mark.parametrize("figure", ["fig2a", "fig2b"])
def test_statistics_match_reference(figure_statistics, figure):
    """Figure 2a/2b statistics match tests/reference_stats.json."""
    reference = load_reference_stats()[figure]
    computed = figure_statistics[figure]

    assert np.allclose(computed["means"], reference["means"],
                       atol=STATS_ATOL, rtol=0)
    assert np.allclose(computed["stds"], reference["stds"],
                       atol=STATS_ATOL, rtol=0)

    assert len(computed["ttests"]) == len(reference["ttests"])
    for computed_row, reference_row in zip(computed["ttests"],
                                           reference["ttests"]):
        assert computed_row["name"] == reference_row["name"]
        for key in ("mean_a", "mean_b", "mean_diff", "t_stat", "p_raw",
                    "p_holm"):
            assert abs(computed_row[key] - reference_row[key]) <= STATS_ATOL, \
                f"{figure}/{computed_row['name']}/{key}"
        assert bool(computed_row["significant"]) == \
            bool(reference_row["significant"])


# ---------------------------------------------------------------------------
# 3. Figure scripts produce their PNGs
# ---------------------------------------------------------------------------
FIGURE_PNGS = {
    "fig2a_articulatory_correlations.py":
        ["fig2a_articulatory_correlations.png"],
    "fig2b_formant_correlations.py":
        ["fig2b_formant_correlations.png"],
    "fig3_formant_trajectories.py":
        ["fig3_formant_trajectories.png"],
    "figs1a_ctw_paths_condition_a.py":
        ["figs1a_ctw_paths_condition_a.png"],
    "figs1b_ctw_paths_condition_b.py":
        ["figs1b_ctw_paths_condition_b.png"],
    "figs3_articulatory_param_trajectories.py":
        ["figs3_articulatory_param_trajectories_condA.png",
         "figs3_articulatory_param_trajectories_condB.png"],
    "figs4_ctw_model15_on_model12.py":
        ["figs4_ctw_model15_on_model12.png"],
    "figs5_ctw_alignment_three_way.py":
        ["figs5_ctw_alignment_three_way.png"],
}


@pytest.mark.parametrize("script", list(FIGURE_PNGS))
def test_figure_script_produces_png(script):
    """Each figure script runs and writes a non-empty PNG."""
    import os

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    # The scripts write into results/figures/ at the repository root
    # (git-ignored regeneration space).
    result = subprocess.run(
        [sys.executable, str(FIGURES_DIR / script)],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)

    assert result.returncode == 0, f"{script} exited with {result.returncode}"
    for png in FIGURE_PNGS[script]:
        path = REPO_ROOT / "results" / "figures" / png
        assert path.exists(), f"{script} did not produce {png}"
        assert path.stat().st_size > 0, f"{png} is empty"

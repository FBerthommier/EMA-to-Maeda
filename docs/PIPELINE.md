# End-to-end data pipeline

This document maps every dependency between the data sets of the
repository, from the raw corpus to the figures, the syllable synthesis and
the videos. Each arrow names the script that implements it. The master
driver `run_all.py` chains all the stages:

```bash
python run_all.py --stage all      # full chain, in order
python run_all.py --stage data     # only the mat/wav -> tracks stage
python run_all.py --stage ctw      # only the CTW alignment stage
python run_all.py --stage figures  # only the figure scripts
python run_all.py --stage synthesis
python run_all.py --stage videos   # requires ffmpeg
```

## Dependency graph

```
data/mat (VV_*.mat, raw EMA)      data/wav (vv_*.wav, audio)   data/maeda (P0*.npy)
        |                                 |                          |
        |  src/batch_parameters.py        |                           |
        v                                 |                           |
   copyparam (CP_*.npy) ------------------|---------------------------+
        |                                 |                           |
        |  |--------------------------------------------------|     |
        |  |  src/make_reference_tracks.py                     |     |
        |  |  (VLAM synthesis driven by the reference          |     |
        |  |   parameters and the recorded energy envelope)    |     |
        |  v                                                  v     |
        |  copywav (CW_*.wav)   copyformants (CF_*.npy)        |     |
        |                                                      |     |
        |     src/ctw_condition_a.py  (model P0<X>,            |     |
        |                               matching reference)    |     |
        |         |--------------------------------------------+     |
        |         v                                                  |
        |     paramodelctw (PM_*.npy) --src/ctw_condition_a.py--> modelctw (MPF_*.npy)
        |                                                            ^
        |     src/ctw_condition_b.py  (random model P0<X2>,          |
        |         |                    seed 126, as figs1b)          |
        |         v                                                  |
        |     paramodelctwrandom (PMR_*.npy) --src/ctw_condition_b.py
        |                                          (VLAM conversion, shared with
        |                                           video/vlam_audio_maker_v2.py)
        v                                                            |
   [copyparam, copyformants, modelctw, modelctwrandom, maeda] --------+
        |
        |  figures/*.py  (8 scripts -> 9 PNGs, incl. the Fig. 2a/2b statistics)
        v
   results/figures/*.png

   synthesis/run_synthesis.py  (Tau-model text -> syllable corpus, no data input)
   video/  (copyparam + modelctwrandom + data/wav -> CondA/CondB/CondRef.mp4
            -> mixARef/mixBRef/mixAB/mixABRef.mp4, ffmpeg required)
```

## Arrow by arrow

| # | From | To | Script | Notes |
|---|---|---|---|---|
| 1 | `data/mat/VV_*.mat` | `copyparam/CP_*.npy` | `src/batch_parameters.py` | 14-channel EMA reduced to 6 Maeda-like parameters, decimated (5:1), 8 Hz zero-phase low-pass, z-normalized, renormalized with the statistics of model `P0<X>`. ~0.5 s per stimulus. |
| 2 | `copyparam/CP_*.npy` + `data/wav/vv_*.wav` | `copywav/CW_*.wav` + `copyformants/CF_*.npy` | `src/make_reference_tracks.py` | VLAM/LPC synthesis of the reference driven by the recorded energy envelope (Hilbert + 8 Hz low-pass, 0.1 threshold). CF_ are the formants of the reference-driven VLAM synthesis, i.e. the reference side of the Figure 2b comparison analyzed with the same VLAM analyzer as the model side. Original provenance: `SynthsylShort/VLAMaudmaker.py` v1 (lines 1255-1289); same computation as `video/vlam_audio_maker.py`. ~40 s per stimulus. |
| 3 | `copyparam/CP_*.npy` + `data/maeda/P0<X>.npy` | `paramodelctw/PM_*.npy` | `src/ctw_condition_a.py` | Condition A: model `P0<X>` CTW-aligned onto the natural parameters (matching model), 8 Hz zero-phase smoothing. The alignment itself is the computation displayed by `figures/figs1a_ctw_paths_condition_a.py` (which does not save it). ~4 s per stimulus. |
| 4 | `paramodelctw/PM_*.npy` | `modelctw/MPF_*.npy` | `src/ctw_condition_a.py` | Parameter-to-formant conversion through VLAM (Hyoid zeroed), dictionary `{'fval', 'Pv'}`; shared converter `src/formant_conversion.py` (identical to `video/vlam_audio_maker_v2.py`). ~35 s per stimulus. |
| 5 | `copyparam/CP_*.npy` + `data/maeda/P0<X2>.npy` (X2 random, seed 126) | `paramodelctwrandom/PMR_*.npy` | `figures/figs1b_ctw_paths_condition_b.py` and `src/ctw_condition_b.py` | Condition B: random model X2 != X; the seeded pairing (seed 126) is replicated call for call by `src/ctw_condition_b.py::pick_random_models`. ~4 s per stimulus. |
| 6 | `paramodelctwrandom/PMR_*.npy` | `modelctwrandom/MPFR_*.npy` | `src/ctw_condition_b.py` | Same converter as arrow 4. ~35 s per stimulus. |
| 7 | `copyparam`, `copyformants`, `modelctw`, `modelctwrandom`, `maeda` | `results/figures/*.png` | `figures/*.py` (8 scripts) | 9 PNGs (Fig. S3 has two panels). Figures 2a/2b also print the statistical analyses (correlations, paired t-tests, Holm correction) recorded in `tests/reference_stats.json`. |
| 8 | (text input) | `results/synthesis/*` | `synthesis/run_synthesis.py` | Tau-model articulatory synthesis of short syllable sequences; reads no data files. |
| 9 | `copyparam/CP_2012.npy`, `modelctwrandom/MPFR_2012.npy`, `data/wav/vv_2012.wav` | `results/videos/*.mp4` | `video/vlam_video_maker.py` + 4 compositing scripts | Renders CondRef/CondA/CondB (XVID AVI at 100 fps, ffmpeg audio mux), then the overlays `mixARef`, `mixBRef`, `mixAB` and the side-by-side `mixABRef`. |

`data/mat`, `data/wav` and `data/maeda` are sources (no generator);
`data/videos` holds the archived MP4s (regenerated by the video stage
into `results/videos/`).

## Where the outputs are written

All generators write under a configurable root (default
`results/regenerated/data`, mirroring the `data/` tree); pass
`--out-root data` to the `src/` generators to rewrite the canonical
locations instead. The archived `data/` tree is therefore read-only for
`run_all.py`, and `tests/test_reproducibility.py` compares the
regenerated values against the archived files.

One documented exception: `figures/figs1b_ctw_paths_condition_b.py` (a
validated figure script, left untouched) re-saves the `PMR_*.npy` files
it computes into `data/paramodelctwrandom/`. The computation is
deterministic (seed 126) and byte-identical to the archive — this is
asserted by the test suite — so the archived tree is effectively
unchanged after a run.

## Numerical reproducibility

Two levels of equality are tested (see `tests/test_reproducibility.py`):

1. **Chain identity** — feeding the *archived* inputs (e.g. `CP_*.npy`)
   to the regenerated generators reproduces the archived outputs to
   machine precision: aligned parameters match to ~5e-14, formant tracks
   to ~1e-11 (parameter units) / ~1e-12 Hz, and the synthesized
   waveforms to at most 1 least-significant bit of the archived 16-bit
   PCM files. This proves the generators are methodologically identical
   to the originals.

2. **End-to-end drift** — restarting from the raw `data/mat/` corpus,
   the regenerated `CP_*.npy` deviate from the archive by at most
   ~2e-5 (documented tolerance 5e-5). This drift comes from library
   versions (the original `copyparam` was produced by the same code on
   an older NumPy/SciPy stack; the EMA reduction chain works on
   float32 data), not from a methodological difference — the original
   `SynthsylShort/batch_param.py` and `src/batch_parameters.py` are
   line-for-line identical. Downstream, the reference formants (CF_)
   stay within ~3e-3 Hz and the reference waveforms within 1 LSB, but
   the CTW warping path is sensitive to these ~1e-5 input
   perturbations: in near-tie regions of the cost matrix the discrete
   path can shift, so a few frames of the aligned tracks deviate by up
   to ~0.3 parameter units and the corresponding formant tracks by up
   to ~100 Hz at those frames, while the vast majority of frames stay
   identical or near-identical and the tracks remain visually
   indistinguishable. The per-artifact tolerances (max deviation plus a
   tight bound on the fraction of affected frames) are set and
   commented in `tests/test_reproducibility.py`.

## Stage runtimes (63 stimuli, one core)

Measured on the reference machine (Python 3.14, NumPy 2.5, SciPy 1.18,
tslearn 0.9); the full `--stage all` chain completes in about 2 hours:

| Stage | Approximate duration |
|---|---|
| `data` | ~45 min (dominated by the VLAM synthesis, ~40 s/stimulus) |
| `ctw` | ~76 min (two conditions x (CTW + VLAM conversion ~35 s/stimulus)) |
| `figures` | ~1 min (the 126 CTW alignments of S1a/S1b are fast after the numba warm-up of the first call) |
| `synthesis` | ~15 s |
| `videos` | ~1 min (matplotlib frame rendering, ffmpeg mux/composites) |

The pytest suite (`tests/test_reproducibility.py`) runs in ~20 min by
subsampling 5 stimuli and re-running the figure scripts.

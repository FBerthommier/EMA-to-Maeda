# Script mapping: legacy `SynthsylShort/` to this repository

| Original script | New script | Article figure | Description |
|---|---|---|---|
| `SynthsylShort/Figure2a.py` | `figures/fig2a_articulatory_correlations.py` | Figure 2a | Distribution of correlations between reference EMA parameters and CTW-aligned Maeda-model parameters (conditions A vs B), violin plot + paired t-tests with Holm correction. |
| `SynthsylShort/Figure2b.py` | `figures/fig2b_formant_correlations.py` | Figure 2b | Same analysis as Figure 2a, for formants F1-F3. |
| `SynthsylShort/Figure3.py` | `figures/fig3_formant_trajectories.py` | Figure 3 | Example formant trajectory comparison (CS vs CTW) for one utterance. |
| `SynthsylShort/FigureS1a.py` | `figures/figs1a_ctw_paths_condition_a.py` | Supplement Fig. S1a | CTW warping paths for all utterances, condition A. |
| `SynthsylShort/FigureS1b.py` | `figures/figs1b_ctw_paths_condition_b.py` | Supplement Fig. S1b | CTW warping paths with random models, condition B (also saves aligned parameters to `data/paramodelctwrandom/`). |
| `SynthsylShort/FigureS3.py` | `figures/figs3_articulatory_param_trajectories.py` | Supplement Fig. S3 | Articulatory parameter trajectories (CS vs CTW) for one utterance, conditions A and B. |
| `SynthsylShort/FigureS4.py` | `figures/figs4_ctw_model15_on_model12.py` | Supplement Fig. S4 | CTW alignment of Maeda Model15 onto Model12 (correlations before/after alignment). |
| `SynthsylShort/FigureS5.py` | `figures/figs5_ctw_alignment_three_way.py` | Supplement Fig. S5 | Same as S4 with the raw Model15 also superposed (three-way comparison). |
| `SynthsylShort/batch_param.py` | `src/batch_parameters.py` | (data preparation) | Batch generation of `data/copyparam/`: reduces 14-channel EMA (.mat) to 6 Maeda-like parameters, decimates, filters, and renormalizes with model statistics. Placed in `src/` because it is a data-preparation utility, not a figure script. Optional `--out-root` redirects the outputs (default: the canonical `data/copyparam/`). |
| `SynthsylShort/superpose2.py` | `video/superpose_conditions_a_b.py` | (video) | Superposition of the condition A and condition B articulation videos (`mixAB.mp4`). |
| `SynthsylShort/superpose2b.py` | `video/superpose_condition_b_reference.py` | (video) | Superposition of condition B over the reference video (`mixBRef.mp4`). |
| `SynthsylShort/superpose.py` *(lost)* | `video/superpose_condition_a_reference.py` | (video) | Superposition of condition A over the reference video (`mixARef.mp4`). The original script was not preserved; rebuilt by symmetry with `superpose2b.py` (hue shift omitted). |
| `SynthsylShort/sidebyside.py` | `video/side_by_side_composites.py` | (video) | Horizontal side-by-side composition of `mixARef.mp4` and `mixBRef.mp4` (`mixABRef.mp4`). |

## New pipeline scripts (end-to-end reproducibility)

The following scripts complete the regeneration chain so that every
intermediate data set of `data/` has a generator (see `docs/PIPELINE.md`
for the full dependency graph, and `run_all.py` for the master driver):

| New script | Generates | Notes |
|---|---|---|
| `run_all.py` | (master driver) | Chains the stages `data -> ctw -> figures -> synthesis -> videos` (`--stage all`); idempotent, logged to `results/logs/run_all.log`. |
| `src/make_reference_tracks.py` | `data/copywav/CW_*.wav`, `data/copyformants/CF_*.npy` | Same computation as `video/vlam_audio_maker.py` (original `VLAMaudmaker.py` v1, lines 1255-1289); writes to a configurable `--out-root` (default `results/regenerated/data`) with the archived PCM_16 wav subtype. |
| `src/ctw_condition_a.py` | `data/paramodelctw/PM_*.npy`, `data/modelctw/MPF_*.npy` | Condition A alignment computed by `figures/figs1a_*.py` (which only plots it), plus the parameter-to-formant conversion. |
| `src/ctw_condition_b.py` | `data/paramodelctwrandom/PMR_*.npy`, `data/modelctwrandom/MPFR_*.npy` | Replicates the seed-126 random pairing of `figures/figs1b_*.py` call for call, plus the conversion (same as `video/vlam_audio_maker_v2.py`). |
| `src/ctw_tracks.py` | (shared helpers) | Model/natural loaders and the CTW + zero-phase smoothing step common to both conditions. |
| `src/formant_conversion.py` | (shared helper) | (7, T) aligned parameters -> `{'fval', 'Pv'}` MPF/MPFR dictionary through VLAM (Hyoid zeroed). |
| `tests/test_reproducibility.py` | (tests) | pytest suite comparing the regenerated intermediates and the Figure 2a/2b statistics (`tests/reference_stats.json`, produced by `tests/generate_reference_stats.py`) against the archives. |

Wiring notes on existing scripts (numerical behavior unchanged):
`src/batch_parameters.py` gained an optional `--out-root`; the four
`video/superpose_*.py` / `video/side_by_side_composites.py` compositing
scripts gained optional `--input-dir` / `--output-dir` / `--ffmpeg`
flags (defaults unchanged); `video/vlam_video_maker.py` had its audio
input fixed from `VV_2012.wav` to the actual `vv_2012.wav` file name
(case-sensitive on Linux) and its `--ffmpeg` flag now actually reaches
the audio muxing step (a mux failure aborts with a non-zero exit).

A compiled usage manual of the refactored functions — CLI commands,
importable API of every module, reproducibility guarantees — is provided
as `docs/usage_manual.pdf` (LaTeX source `docs/usage_manual.tex`).

## Video compositing chain (usage order)

The archived MP4s were originally assembled manually with the
`superpose*.py` / `sidebyside.py` wrappers. The dependency order is:

```
vlam_video_maker (copyparam)      -> CondA.mp4      (and CondRef.mp4, no overlay)
vlam_video_maker (modelctwrandom) -> CondB.mp4
vlam_audio_maker / _v2            -> CW_*.wav, CF_*.npy, MPFR_*.npy (parameters/formants)
superpose_condition_a_reference   : CondRef + CondA -> mixARef.mp4
superpose_condition_b_reference   : CondRef + CondB -> mixBRef.mp4
superpose_conditions_a_b          : CondA  + CondB  -> mixAB.mp4
side_by_side_composites           : mixARef | mixBRef -> mixABRef.mp4
```

See `video/README.md` for the exact commands. The authoritative set of
elementary videos was the `Videos-non-anonymous` working directory;
`data/videos/` now holds exactly that set (verified by MD5). The Zenodo
archive keeps only the final `mixABRef.mp4` at its root (clickable
from the supplement PDF), the variant used for supplement figure S2b having
been kept separately as `mixABRef-S2b.mp4`. Historical note: the
`CondA.mp4` / `CondB.mp4` files found in the legacy `SynthsylShort` folder
(and initially recopied here) were mislabeled byte-identical copies of the
reference video (`CondRef.mp4`); they were replaced in `data/videos/` by
the genuine condition A and B videos of `VideoMaker` /
`Videos-non-anonymous`.

## Shared modules extracted into `src/`

- `src/correlations.py` — `compute_correlation_simple` (manual Pearson correlation, ddof=1).
- `src/ctw.py` — `apply_ctw_path` (CTW alignment with gap interpolation), `zero_phase_lowpass_filter`.
- `src/stats_tests.py` — `paired_ttests_holm` (paired t-tests + Holm correction, used by Figures 2a/2b).
- `src/plots.py` — `plot_articulatory_parameters` and `plot_ctw_path` matplotlib helpers.

## Provenance of the intermediate data

- `data/copyparam/CP_*.npy` — generated from `data/mat/` by `src/batch_parameters.py`.
- `data/copywav/CW_*.wav` and `data/copyformants/CF_*.npy` — generated by the
  VLAM reference-synthesis chain (original `VLAMaudmaker.py` v1, refactored as
  `video/vlam_audio_maker.py`): envelope of the natural recording
  (`data/wav/vv_*.wav`) drives an LPC/VLAM synthesis of the reference
  parameters; the synthetic waveform is stored as CW_* and its F1–F3 tracks as
  CF_*. Both sides of the Figure 2b comparison are thus analyzed with the same
  VLAM analyzer.
- `data/paramodelctwrandom/PMR_*.npy` — generated by
  `figures/figs1b_ctw_paths_condition_b.py` (seed 126) and identically by
  `src/ctw_condition_b.py` (which replicates the seeded pairing call for
  call and writes to a configurable output root).
- `data/paramodelctw/PM_*.npy` and `data/modelctw/MPF_*.npy` — generated
  by `src/ctw_condition_a.py` (CTW of the matching model `P0<X>` onto the
  reference, zero-phase smoothing, then VLAM formant conversion).
- `data/modelctwrandom/MPFR_*.npy` — generated by
  `src/ctw_condition_b.py` (same VLAM conversion as
  `video/vlam_audio_maker_v2.py`).

## Data layout

All scripts resolve data relative to the repository root via `Path`:

`copyparam/` -> `data/copyparam/`, `copyformants/` -> `data/copyformants/`,
`Maeda/` -> `data/maeda/`, `modelctw/` -> `data/modelctw/`,
`modelctwrandom/` -> `data/modelctwrandom/`, `paramodelctw/` -> `data/paramodelctw/`,
`paramodelctwrandom/` -> `data/paramodelctwrandom/`, `copywav/` -> `data/copywav/`,
`wav/` -> `data/wav/`, `mat/` -> `data/mat/`.

Figures are saved to `results/figures/` (300 dpi); videos to `results/videos/`.

## `SynthsylShort/synthSYLVcourtes.py` -> `synthesis/` package

Legacy script (2676 lines, French comments): interactive articulatory
synthesizer for short syllable sequences. Input words (vowels `u o O a è é i y E`,
consonants `b d g`, syllables separated by dots) are parsed into CV skeletons;
Tau-model trajectories move 7 articulatory parameters through polar targets;
the VLAM (Maeda-style) sagittal model converts the parameters into a
29-section area function; a lossless-tube model (nraph3/ICP heritage) yields
the spectrum and F1-F3; an LPC source-filter chain synthesizes the audio.
Outputs: `essai.wav`, `essai.npy` (formants + parameters), `output.avi`
(vocal-tract animation), plus parameter/formant/spectrogram plots.

| New module | Contents |
|---|---|
| `synthesis/__init__.py` | Package docstring and provenance. |
| `synthesis/config.py` | All constants: sampling rate (20 kHz), Tau-model settings (dur=16, K=Kvoy=Kpause=1000, Pexp=2, valrect=0.75), vowel/consonant inventories and polar targets, movement-circle geometry `CO`, articulator tables `ART`/`ART1`, tube-model physical constants, repo/data/results paths. |
| `synthesis/acoustic_model.py` | Tube model: `softrect`, `poly2rc`, `levinson_durbin`, `Hfreq2lpc`, `spectrelec`, `nraph_oral` (Newton pole search, ex nraph3.ftn ICP Grenoble), `aire2spectre_oral`/`aire2spectre_cor_oral`, `vtn2frm_ftr_oral`, `freqevalNN` (articulators -> F1-F3 + spectrum), `synthpause`. |
| `synthesis/lpc_synthesis.py` | Audio synthesis: `gen_src_3` (glottal pulse source, L. Girin GIPSA-lab 2009), `f_lpc_exc2sig` (time-varying LPC lattice filter), `synthfen` (word energy envelope), `synthsimpleSYL`, `synthsimpleWORDfen`, default F0 contour with jitter. |
| `synthesis/vlam.py` | VLAM/Maeda articulatory model: `initVLAMLength`, `vlam2009NN` (sagittal contours + area function), `showgui`. |
| `synthesis/trajectories.py` | Tau-model movement generation: `arc`, `arcplot`, `arcplotV`, `makeloop`, `makeloopplot`, `boucle`, `boucleplot`, consonant target functions (`consvalD/G/D2/G2`), syllable `parse`. |
| `synthesis/syllables.py` | Syllable dict construction: `syllable_types`, `build_syllable` (the type-dispatch block of legacy main()). |
| `synthesis/pipeline.py` | End-to-end pipeline: `synthsyl`, `synthwordfen`, `synthesize_text` (word/syllable loop of legacy main()). |
| `synthesis/audio_io.py` | `play` (sounddevice, gated by `config.PLAYBACK_ENABLED`), `save_wav` (int16 wav), `save_parameters` (.npy dict fval/Pv/dur). |
| `synthesis/plotting.py` | `plot_Pv`, `plot_formants`, `spectreplot` (dead formant-overlay branch removed). |
| `synthesis/video.py` | `playVLAMvid`: matplotlib frames -> OpenCV AVI. |
| `synthesis/run_synthesis.py` | CLI entry point: `--text`, `--condition`, `--stimulus`, `--output-dir`, `--stem`, `--play`, `--no-video`, `--no-plots`, `--no-spectrogram`; interactive prompt if `--text` omitted. |

Removed as dead code (defined but never called): `voysynth`, `vlam2009NN_old`,
`boucleword_old`, `bouclewordplot_old`, `playVLAM` (GIF exporter; also ignored
its output-path argument), `lpc_formants`/`extract_formants` (only used in an
`if 0:` block).

Output files keep their legacy names (`essai.wav`, `essai.npy`, `output.avi`)
but are written to `results/synthesis/` by default; `--stimulus`/`--condition`
prefix the output stem.

`synthSYLVcourtes.py` reads no external data files; the `data/` subdirectories
listed above belong to the other scripts of this table.

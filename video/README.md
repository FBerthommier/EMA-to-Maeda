# Video stimuli — animated articulatory speech (VLAM)

This folder reproduces the MP4 videos accompanying the Interspeech 2026
paper-2479 (articulatory synthesis with VLAM):

- **CondA.mp4** — condition A: animated midsagittal articulation driven
  by the reference articulatory trajectories (`data/copyparam/`), with
  the synthesized speech track.
- **CondB.mp4** — condition B: same animation driven by the converted
  CTW-random trajectories (`data/paramodelctwrandom/` /
  `data/modelctwrandom/`).
- **CondRef.mp4** — reference video (raw animation, no condition
  overlay); needed as the background for the composites below.
- **mixARef.mp4 / mixBRef.mp4** — condition A / condition B overlaid on
  the reference with 40% transparency.
- **mixAB.mp4** — condition A and condition B overlaid on each other.
- **mixABRef.mp4** — final side-by-side composite
  (mixARef.mp4 | mixBRef.mp4).

## Contents

| File | Purpose |
|---|---|
| `video_utils.py` | Shared library: VLAM articulatory forward model, vocal-tract acoustics (transfer function + formants), LPC synthesizer, envelope extraction, parameter I/O, matplotlib/OpenCV rendering, ffmpeg muxing. |
| `vlam_audio_maker.py` | Version 1: full LPC articulatory synthesis (waveform + formants) for the `copyparam` corpus (`CP_*` → `CW_*.wav`, `CF_*.npy`). |
| `vlam_audio_maker_v2.py` | Version 2: formant/trajectory conversion only for the CTW-random corpus (`PMR_*` → `MPFR_*.npy` dictionaries). |
| `vlam_video_maker.py` | Renders the midsagittal animation (AVI, 100 fps) and muxes the speech audio into the final MP4. |
| `superpose_condition_a_reference.py` | CondA overlaid on CondRef (40% alpha) → `mixARef.mp4`. |
| `superpose_condition_b_reference.py` | CondB (hue-shifted green→blue, 40% alpha) overlaid on CondRef → `mixBRef.mp4`. |
| `superpose_conditions_a_b.py` | CondA (blue tint, 40% alpha) overlaid on CondB → `mixAB.mp4`. |
| `side_by_side_composites.py` | mixARef and mixBRef stacked horizontally → `mixABRef.mp4`. |

Refactored from `SynthsylShort/VLAMaudmaker.py`, `SynthsylShort/VLAMaudmaker2.py`,
`SynthsylShort/VLAMvidmaker2.py`, `SynthsylShort/superpose2.py`,
`SynthsylShort/superpose2b.py` and `SynthsylShort/sidebyside.py`. The original
`superpose.py` (condition A over reference) was not preserved; it was
reconstructed by symmetry with `superpose2b.py`. Numerical/visual behavior is
unchanged (same codecs: XVID AVI at 100 fps for the raw renders, libx264 MP4
at 25 fps for the composites; same 20 kHz synthesis sampling rate).

## Requirements

- Python 3.9+ with: `numpy`, `scipy`, `matplotlib`, `opencv-python` (`cv2`),
  `soundfile` (audio reading/writing; falls back to `scipy.io.wavfile` in the
  legacy scripts).
- **ffmpeg** on the `PATH` (or edit the `FFMPEG` constant in
  `video_utils.py`). Only used to mux the audio track into the MP4.

```
pip install numpy scipy matplotlib opencv-python soundfile
```

Run the scripts from this `video/` directory (they import `video_utils`
as a sibling module).

## End-to-end reproduction

The whole chain below is automated by the repository-root driver
(`python run_all.py --stage videos`): it locates ffmpeg (PATH or the
bundled `imageio-ffmpeg` binary), renders the three elementary videos
into `results/videos/` and runs the compositing steps with
`--input-dir/--output-dir results/videos`, so the archived `data/videos/`
tree is never touched. The manual commands are given for reference.

All commands below are run from `Github/video/`.

### 1. Synthesize the audio for condition A (version 1 audio maker)

```bash
python vlam_audio_maker.py
```

- Reads: `data/copyparam/CP_<Y>0<X>.npy` and `data/wav/vv_<Y>0<X>.wav`
  for X in 10..16, Y in 2..10 (63 stimuli).
- Writes: `results/videos/CW_<name>.wav` (20 kHz synthesized speech) and
  `results/videos/CF_<name>.npy` (F1–F3 trajectories).
- Optional flags: `--fs-in`, `--fs-out` (default 100 Hz in/out, no
  resampling by default), `--out-dir`.
- Runtime: a few minutes per stimulus (the Newton–Raphson formant
  search dominates). Each stimulus is a short syllable; the resulting
  audio track duration equals the duration of the corresponding
  `data/wav/vv_<name>.wav` recording (about 1 s each).

### 2. Convert the CTW-random trajectories for condition B (version 2 audio maker)

```bash
python vlam_audio_maker_v2.py
```

- Reads: `data/paramodelctwrandom/PMR_<Y>0<X>.npy` (same X/Y ranges).
- Writes: `results/videos/MPFR_<name>.npy` — dictionaries
  `{'fval': formants, 'Pv': parameters}` (the legacy equivalent of the
  `modelctwrandom/MPFR_*.npy` files).
- No waveform synthesis: this step only computes formants and stores the
  converted trajectories used to animate condition B.

### 3. Render the video and add the audio (video maker)

Condition A (reproduces **CondA.mp4**):

```bash
python vlam_video_maker.py --source copyparam --mp4 ../results/videos/CondA.mp4
```

Condition B (reproduces **CondB.mp4**):

```bash
python vlam_video_maker.py --source modelctwrandom --mp4 ../results/videos/CondB.mp4
```

Each run:

- Renders all articulatory frames with matplotlib (100 dpi, 6x4 inch,
  XVID AVI at 100 fps) → `results/videos/vlm_output.avi`.
- Muxes the speech audio `data/wav/VV_2012.wav` with ffmpeg → the final
  MP4 (about 1 s duration, one audio/video stream, video stream copied
  without re-encoding).

**Compositing chain.** The archived MP4s were originally assembled
manually with the `superpose*.py` / `sidebyside.py` ffmpeg wrappers.
The full chain is now scripted, in dependency order:

```bash
# Background: the reference video must exist first
python vlam_video_maker.py --source copyparam    --mp4 ../results/videos/CondRef.mp4
# Single-condition videos
python vlam_video_maker.py --source copyparam      --mp4 ../results/videos/CondA.mp4
python vlam_video_maker.py --source modelctwrandom --mp4 ../results/videos/CondB.mp4
# Composites (each needs the previous outputs in data/videos/)
python superpose_condition_a_reference.py   # CondRef + CondA -> mixARef.mp4
python superpose_condition_b_reference.py   # CondRef + CondB -> mixBRef.mp4
python superpose_conditions_a_b.py          # CondA + CondB   -> mixAB.mp4
python side_by_side_composites.py           # mixARef | mixBRef -> mixABRef.mp4
```

Copy each produced video into `data/videos/` (or adjust `INPUT_DIR`) before
running the composites that consume it. In the published archive,
`mixABRef-S2b.mp4` is the variant of `mixABRef.mp4` used for supplement
figure S2b.

**Archived set in `data/videos/`.** The seven MP4s shipped in
`data/videos/` are the authoritative working-directory set
(`Videos-non-anonymous`), verified by MD5: `CondRef.mp4` (raw reference
render), the genuine `CondA.mp4` and `CondB.mp4` (distinct renders; the
`CondA/CondB` files first recopied from the legacy `SynthsylShort` folder
were mislabeled byte-identical copies of `CondRef.mp4` and have been
replaced), and the composites `mixARef`, `mixBRef`, `mixAB`, `mixABRef`.
The composites shipped here are one consistent assembly run; re-running
the chain reproduces the same content but not the same encoded bytes
(encoder versions differ).

**mixAB.mp4** can also be assembled directly with ffmpeg:

```bash
ffmpeg -y -i CondA.mp4 -i CondB.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? -shortest mixAB.mp4
```

## Notes

- To use a different utterance than `2012`, edit the `UTTERANCE`
  constant in `vlam_video_maker.py`.
- Output locations follow the repository layout: inputs under
  `data/`, videos and audio under `results/videos/`. The legacy
  scripts wrote the synthesized audio back into the source folders;
  the refactored versions keep `data/` read-only.

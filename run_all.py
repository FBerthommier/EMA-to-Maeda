"""Master reproduction pipeline of the article and its supplement.

This driver regenerates every numerical result of the study from the raw
data (``data/mat/``, ``data/wav/``, ``data/maeda/``) through the
intermediate tracks to the figures, the syllable synthesis corpus and the
MP4 stimulus videos. Every generator writes under
``results/regenerated/`` (same tree structure as ``data/``); the archived
``data/`` directory is read-only for the pipeline, and the pytest suite
(``tests/test_reproducibility.py``) compares the regenerated values
against the archived files.

Stages (run with ``--stage all`` for the full chain, in this order):

    data      data/mat + data/wav -> copyparam, copywav, copyformants
    ctw       copyparam           -> paramodelctw(random), modelctw(random)
    figures   archived data/      -> the 9 article/supplement PNGs
    synthesis Tau-model text      -> syllable corpus (results/synthesis)
    videos    copyparam/modelctwrandom + data/wav -> the MP4 video chain

Each stage is idempotent (already-present outputs are skipped; use
``--force`` to recompute) and logged to the console and to
``results/logs/run_all.log``.

Randomness is fully seeded inside the generators themselves: the
condition B model pairing uses seed 126 (as ``figures/figs1b_*.py``) and
the LPC synthesis reseeds numpy (seed 0) per stimulus, so repeated runs
are deterministic on a fixed environment.

Note: ``figures/figs1b_ctw_paths_condition_b.py`` also re-saves the
PMR_*.npy files it computes into ``data/paramodelctwrandom/``; the
regeneration is deterministic and byte-identical to the archive (this is
asserted by the pytest suite), so the archived tree is effectively
unchanged.

Usage:
    python run_all.py --stage all
    python run_all.py --stage data --force
    python run_all.py --stage figures
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
DEFAULT_OUT_ROOT = RESULTS_DIR / "regenerated" / "data"

# One entry per figure script: (script, [produced PNG names]).
FIGURE_SCRIPTS = [
    ("fig2a_articulatory_correlations.py",
     ["fig2a_articulatory_correlations.png"]),
    ("fig2b_formant_correlations.py",
     ["fig2b_formant_correlations.png"]),
    ("fig3_formant_trajectories.py",
     ["fig3_formant_trajectories.png"]),
    ("figs1a_ctw_paths_condition_a.py",
     ["figs1a_ctw_paths_condition_a.png"]),
    ("figs1b_ctw_paths_condition_b.py",
     ["figs1b_ctw_paths_condition_b.png"]),
    ("figs3_articulatory_param_trajectories.py",
     ["figs3_articulatory_param_trajectories_condA.png",
      "figs3_articulatory_param_trajectories_condB.png"]),
    ("figs4_ctw_model15_on_model12.py",
     ["figs4_ctw_model15_on_model12.png"]),
    ("figs5_ctw_alignment_three_way.py",
     ["figs5_ctw_alignment_three_way.png"]),
]

# Elementary renders (condition source -> output MP4 name).
VIDEO_RENDERS = [
    ("copyparam", "CondRef.mp4"),
    ("copyparam", "CondA.mp4"),
    ("modelctwrandom", "CondB.mp4"),
]

# Compositing steps (script -> output MP4 name).
VIDEO_COMPOSITES = [
    ("superpose_condition_a_reference.py", "mixARef.mp4"),
    ("superpose_condition_b_reference.py", "mixBRef.mp4"),
    ("superpose_conditions_a_b.py", "mixAB.mp4"),
    ("side_by_side_composites.py", "mixABRef.mp4"),
]

# Demonstration corpus of the synthesis stage (vowels o O a e i y E,
# consonants b d g, dots separate syllables).
SYNTHESIS_TEXTS = ["ba.du abi", "ga.bi.du"]

SYNTHESIS_OUTPUT_DIR = RESULTS_DIR / "synthesis"

LOG = logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Infrastructure: logging and subprocesses
# ---------------------------------------------------------------------------
def setup_logging():
    """Attach a console handler and a file handler to the pipeline logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(console)

    logfile = logging.FileHandler(LOG_DIR / "run_all.log", encoding="utf-8")
    logfile.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(logfile)


def run_command(cmd, cwd=None, extra_path=None):
    """
    Run one pipeline command, streaming its output into the logger.

    Args:
        cmd (list of str): Command and arguments.
        cwd (Path or None): Working directory (default: repository root).
        extra_path (Path or None): Directory prepended to PATH for the
            subprocess (used to expose a bundled ffmpeg binary).

    Raises:
        SystemExit: If the command returns a non-zero exit code.
    """
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"          # Headless matplotlib everywhere.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_path is not None:
        env["PATH"] = str(extra_path) + os.pathsep + env.get("PATH", "")

    LOG.info("$ %s", " ".join(str(c) for c in cmd))
    start = time.perf_counter()
    process = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd if cwd is not None else REPO_ROOT),
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        LOG.info("| %s", line.rstrip())
    process.stdout.close()
    code = process.wait()
    elapsed = time.perf_counter() - start

    if code != 0:
        LOG.error("Command failed with exit code %d after %.1f s: %s",
                  code, elapsed, " ".join(str(c) for c in cmd))
        raise SystemExit(1)
    LOG.info("(done in %.1f s)", elapsed)


def require_dirs(dirs, stage):
    """Stop with a clear message if one of the required inputs is missing."""
    for d in dirs:
        if not Path(d).is_dir():
            LOG.error("[%s] missing input directory: %s -- the repository "
                      "data/ tree seems incomplete.", stage, d)
            raise SystemExit(1)


def locate_ffmpeg():
    """
    Locate an ffmpeg executable.

    Returns:
        tuple: (ffmpeg path or None, directory to prepend to PATH or
        None). Returns (None, None) if no ffmpeg can be found.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found, None
    try:
        import imageio_ffmpeg
        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled.exists():
            return bundled, bundled.parent
    except ImportError:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_data(out_root, force):
    """Regenerate copyparam, then copywav/copyformants, from the raw corpus."""
    require_dirs([REPO_ROOT / "data" / "mat", REPO_ROOT / "data" / "wav",
                  REPO_ROOT / "data" / "maeda"], "data")

    cmd = [sys.executable, REPO_ROOT / "src" / "batch_parameters.py",
           "--out-root", out_root]
    run_command(cmd)

    cmd = [sys.executable, REPO_ROOT / "src" / "make_reference_tracks.py",
           "--copyparam-dir", Path(out_root) / "copyparam",
           "--out-root", out_root]
    if force:
        cmd.append("--force")
    run_command(cmd)


def stage_ctw(out_root, force):
    """Regenerate the condition A and condition B CTW products."""
    copyparam_dir = Path(out_root) / "copyparam"
    if not copyparam_dir.is_dir():
        copyparam_dir = REPO_ROOT / "data" / "copyparam"
        LOG.info("[ctw] no regenerated copyparam found, using the archived "
                 "%s", copyparam_dir)
    require_dirs([copyparam_dir, REPO_ROOT / "data" / "maeda"], "ctw")

    for script in ("ctw_condition_a.py", "ctw_condition_b.py"):
        cmd = [sys.executable, REPO_ROOT / "src" / script,
               "--copyparam-dir", copyparam_dir, "--out-root", out_root]
        if force:
            cmd.append("--force")
        run_command(cmd)


def stage_figures(force):
    """Run the eight figure scripts (nine PNG outputs)."""
    require_dirs([REPO_ROOT / "data" / "copyparam",
                  REPO_ROOT / "data" / "copyformants",
                  REPO_ROOT / "data" / "modelctw",
                  REPO_ROOT / "data" / "modelctwrandom",
                  REPO_ROOT / "data" / "maeda"], "figures")

    figure_dir = RESULTS_DIR / "figures"
    for script, pngs in FIGURE_SCRIPTS:
        if not force and all((figure_dir / p).exists() for p in pngs):
            LOG.info("[figures] %s: PNGs already present, skipped", script)
            continue
        run_command([sys.executable, REPO_ROOT / "figures" / script])


def stage_synthesis(force):
    """Synthesize the demonstration syllable corpus."""
    for text in SYNTHESIS_TEXTS:
        stem = text.replace(" ", "_").replace(".", "-")
        if not force and (SYNTHESIS_OUTPUT_DIR / f"{stem}.wav").exists():
            LOG.info("[synthesis] '%s': output already present, skipped",
                     text)
            continue
        run_command([
            sys.executable, REPO_ROOT / "synthesis" / "run_synthesis.py",
            "--text", text, "--stem", stem,
            "--output-dir", SYNTHESIS_OUTPUT_DIR,
            "--no-video", "--no-plots", "--no-spectrogram",
        ])


def stage_videos(force):
    """Render the elementary MP4s and run the compositing chain."""
    require_dirs([REPO_ROOT / "data" / "copyparam",
                  REPO_ROOT / "data" / "modelctwrandom",
                  REPO_ROOT / "data" / "wav"], "videos")

    ffmpeg, extra_path = locate_ffmpeg()
    if ffmpeg is None:
        LOG.error("[videos] ffmpeg was not found (not on PATH and no "
                  "imageio-ffmpeg package). Install ffmpeg or "
                  "'pip install imageio-ffmpeg' and re-run this stage.")
        raise SystemExit(1)
    LOG.info("[videos] using ffmpeg: %s", ffmpeg)

    video_dir = RESULTS_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for source, mp4_name in VIDEO_RENDERS:
        target = video_dir / mp4_name
        if not force and target.exists():
            LOG.info("[videos] %s already present, skipped", mp4_name)
            continue
        run_command([sys.executable, REPO_ROOT / "video" /
                     "vlam_video_maker.py",
                     "--source", source, "--mp4", target,
                     "--ffmpeg", ffmpeg],
                    cwd=REPO_ROOT / "video", extra_path=extra_path)
        if not target.exists():
            LOG.error("[videos] %s was not produced by the render (audio "
                      "muxing probably failed).", mp4_name)
            raise SystemExit(1)

    for script, mp4_name in VIDEO_COMPOSITES:
        target = video_dir / mp4_name
        if not force and target.exists():
            LOG.info("[videos] %s already present, skipped", mp4_name)
            continue
        run_command([sys.executable, REPO_ROOT / "video" / script,
                     "--input-dir", video_dir, "--output-dir", video_dir,
                     "--ffmpeg", ffmpeg],
                    extra_path=extra_path)


STAGES = {
    "data": lambda args: stage_data(args.out_root, args.force),
    "ctw": lambda args: stage_ctw(args.out_root, args.force),
    "figures": lambda args: stage_figures(args.force),
    "synthesis": lambda args: stage_synthesis(args.force),
    "videos": lambda args: stage_videos(args.force),
}
STAGE_ORDER = ["data", "ctw", "figures", "synthesis", "videos"]


def main():
    """Parse the command line and run the requested stage(s)."""
    parser = argparse.ArgumentParser(
        description="Regenerate the article results from the raw data. "
                    "Generators write under results/regenerated/; the "
                    "archived data/ tree is never modified.")
    parser.add_argument("--stage", required=True,
                        choices=STAGE_ORDER + ["all"],
                        help="Stage to run (or 'all' for the full chain in "
                             "order).")
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT),
                        help="Root of the regenerated data tree (default: "
                             "results/regenerated/data).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute outputs that already exist.")
    args = parser.parse_args()

    setup_logging()
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]

    LOG.info("Pipeline stage(s): %s", ", ".join(stages))
    LOG.info("Regenerated data root: %s", Path(args.out_root).resolve())
    start = time.perf_counter()
    for stage in stages:
        LOG.info("=" * 70)
        LOG.info("Stage '%s' starting", stage)
        stage_start = time.perf_counter()
        STAGES[stage](args)
        LOG.info("Stage '%s' done in %.1f s", stage,
                 time.perf_counter() - stage_start)
    LOG.info("=" * 70)
    LOG.info("All requested stages completed in %.1f s.",
             time.perf_counter() - start)


if __name__ == "__main__":
    main()

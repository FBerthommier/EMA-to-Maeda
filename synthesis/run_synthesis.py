"""Command-line entry point for the articulatory synthesizer.

Refactored from SynthsylShort/synthSYLVcourtes.py (main() and __main__).

Synthesizes short syllable sequences ("SYLV courtes"): each input word is a
string of vowel/consonant characters, syllables separated by dots, e.g.
"ba.du abi". Vowels: u o O a è é i y E. Consonants: b d g.

Outputs (written to --output-dir, default results/synthesis/):
- <stem>.wav  : synthesized audio (20 kHz int16),
- <stem>.npy  : dict with formant trajectories (fval), articulatory
                parameter trajectories (Pv) and segment duration (dur),
- output.avi  : vocal-tract animation (unless --no-video).

Example:
    python -m synthesis.run_synthesis --text "ba.du abi" --stem essai
"""

import argparse
import sys


def _setup_imports():
    """Allow running this file directly (python synthesis/run_synthesis.py)."""
    import os
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_setup_imports()

import matplotlib.pyplot as plt  # noqa: E402

from synthesis import config  # noqa: E402
from synthesis.audio_io import save_parameters, save_wav  # noqa: E402
from synthesis.pipeline import synthesize_text  # noqa: E402
from synthesis.plotting import plot_Pv, plot_formants, spectreplot  # noqa: E402
from synthesis.video import playVLAMvid  # noqa: E402
from synthesis.vlam import initVLAMLength  # noqa: E402


def _resize_console():
    """Resize/move the Windows console (cosmetic only, as in the original)."""
    if sys.platform == "win32":
        import ctypes
        import os
        os.system("mode con: cols=60 lines=5")
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        ctypes.windll.user32.MoveWindow(hwnd, 0, 0, 800, 200, True)


def run_one(Rs, args):
    """Synthesize one input string and write all output files."""
    gui = initVLAMLength(config.INIT_PARAMS, config.INIT_TRACK_LENGTH)

    sig, fval, Pv = synthesize_text(Rs, gui, valrect=config.VALRECT,
                                    dur=config.DUR, cf0=config.CF0)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem

    if args.video and Pv.size:
        playVLAMvid(str(output_dir / config.DEFAULT_VIDEO_NAME), 100,
                    gui, Pv, config.VALRECT)

    if args.plots and Pv.size:
        plot_Pv(Pv,
                labels=config.PARAM_LABELS,
                x_range=(0, Pv.shape[0] - 1),
                y_range=(-4, 5), fig_number=5)
        plot_formants(fval, fig_number=6)

    if len(sig) > 0:
        wav_path = save_wav(sig, output_dir / f"{stem}.wav")
        print(f"Wrote {wav_path}")
        save_parameters({"fval": fval, "Pv": Pv, "dur": config.DUR},
                        output_dir / f"{stem}.npy")
        print(f"Wrote {output_dir / (stem + '.npy')}")
        if args.spectrogram:
            spectreplot(sig, config.FS, 1, Rs, 0)


def make_parser():
    parser = argparse.ArgumentParser(
        prog="synthesis.run_synthesis",
        description="Articulatory synthesis of short syllable sequences "
                    "(refactored from SynthsylShort/synthSYLVcourtes.py). "
                    'Input words use vowels %s and consonants %s, syllables '
                    'separated by dots, e.g. "ba.du abi".'
                    % ("".join(config.VOWELS), "".join(config.CONSONANTS)),
    )
    parser.add_argument("--text", default=None,
                        help='Input words to synthesize (e.g. "ba.du abi"). '
                             "If omitted, the script prompts interactively "
                             "(enter N to stop), like the original script.")
    parser.add_argument("--condition", default=None,
                        help="Optional condition label (A/B); only used to "
                             "prefix the output stem.")
    parser.add_argument("--stimulus", default=None,
                        help="Optional stimulus name; used to prefix the "
                             "output stem.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/synthesis).")
    parser.add_argument("--stem", type=str, default=config.DEFAULT_AUDIO_STEM,
                        help="Output file stem (default: essai, matching the "
                             "original essai.wav / essai.npy outputs).")
    parser.add_argument("--play", action="store_true",
                        help="Play audio during synthesis (off by default).")
    parser.add_argument("--no-video", dest="video", action="store_false",
                        help="Disable the vocal-tract AVI video output.")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Disable the parameter/formant plot windows.")
    parser.add_argument("--no-spectrogram", dest="spectrogram",
                        action="store_false",
                        help="Disable the final spectrogram display.")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)

    if args.play:
        config.PLAYBACK_ENABLED = True

    if args.output_dir is None:
        args.output_dir = config.RESULTS_DIR
    else:
        from pathlib import Path
        args.output_dir = Path(args.output_dir)

    # Output stem prefix from optional stimulus/condition labels.
    prefix = "_".join(p for p in (args.stimulus, args.condition) if p)
    if prefix:
        args.stem = f"{prefix}_{args.stem}"

    _resize_console()
    plt.close("all")

    if args.text is not None:
        run_one(args.text, args)
        return

    # Interactive mode, matching the original input() loop.
    Rs = input("Input words: ")
    while Rs != "N":
        if Rs != "":
            run_one(Rs, args)
        Rs = input("Input words (or N to stop): ").rstrip("\n")


if __name__ == "__main__":
    main()

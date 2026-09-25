"""Build the final side-by-side video comparing conditions A and B.

The two composite videos mixARef.mp4 (condition A + reference) and
mixBRef.mp4 (condition B + reference) are scaled to a height of 480 px
at 25 fps and stacked horizontally. The output is written to
results/videos/mixABRef.mp4.

Inputs are looked up in data/videos/ relative to the repository root.

Original script: SynthsylShort/sidebyside.py
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Path to the ffmpeg executable (adjust if ffmpeg is not on the PATH)
FFMPEG_EXE = "ffmpeg"

# Directory containing the input videos
INPUT_DIR = REPO_ROOT / "data" / "videos"

# Directory where the output video is written
OUTPUT_DIR = REPO_ROOT / "results" / "videos"


def build_filter_complex():
    """
    Build the ffmpeg filter_complex string for the side-by-side layout.

    Returns:
        str: The filter graph: both inputs are scaled to a height of
            480 px (width auto) and stacked horizontally.
    """
    return (
        "[0:v]fps=25,scale=-1:480[mixA];"
        "[1:v]fps=25,scale=-1:480[mixB];"
        "[mixA][mixB]hstack=inputs=2[v]"  # side by side
    )


def run_ffmpeg(inputs, filter_complex, output_path):
    """
    Run ffmpeg to composite input videos into an output video.

    Args:
        inputs (list of str): Input video paths, in ffmpeg input order.
        filter_complex (str): ffmpeg filter graph string.
        output_path (Path): Output video path.
    """
    cmd = [
        FFMPEG_EXE,
        "-y",
        *[arg for path in inputs for arg in ("-i", str(path))],
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    """Compose the side-by-side condition A / condition B video."""
    import argparse
    global FFMPEG_EXE

    parser = argparse.ArgumentParser(
        description="Compose mixABRef.mp4 (mixARef | mixBRef side by side).")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help="Directory containing mixARef.mp4 and "
                             "mixBRef.mp4 (default: data/videos).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for mixABRef.mp4 "
                             "(default: results/videos).")
    parser.add_argument("--ffmpeg", type=str, default=FFMPEG_EXE,
                        help="Path to the ffmpeg executable.")
    args = parser.parse_args()

    FFMPEG_EXE = args.ffmpeg
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / "mixABRef.mp4"

    print("Creating the side-by-side video...")

    filter_complex = build_filter_complex()
    run_ffmpeg([str(input_dir / "mixARef.mp4"), str(input_dir / "mixBRef.mp4")],
               filter_complex, out_video)

    print(f"Side-by-side video created: {out_video}")


if __name__ == "__main__":
    main()

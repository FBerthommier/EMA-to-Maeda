"""Superimpose the condition A and condition B articulation videos.

The two videos are scaled to a height of 480 px at 25 fps, the second
one is tinted toward blue with 40% transparency, and the two are
overlaid. The output is written to results/videos/mixAB.mp4.

Inputs are looked up in data/videos/ relative to the repository root.
Note: the input order and filter-graph labels reproduce the original
script exactly (CondA.mp4 is ffmpeg input 0, CondB.mp4 is input 1).

Original script: SynthsylShort/superpose2.py
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
    Build the ffmpeg filter_complex string for the overlay.

    Returns:
        str: The filter graph (labels follow the original script):
            1) scale input 1 to a height of 480 px at 25 fps;
            2) scale input 0, convert to RGBA and apply a light blue
               tint with 40% alpha;
            3) overlay the second stream on the first.
    """
    return (
        # 1) Resize input 1
        "[1:v]fps=25,scale=-1:480[condA];"
        # 2) Resize input 0 and apply a light blue tint with transparency
        "[0:v]fps=25,scale=-1:480,format=rgba,"
        "colorchannelmixer=rr=0.3:gg=0.3:bb=1.0:aa=0.40[condB];"
        # 3) Overlay
        "[condA][condB]overlay=0:0[v]"
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
    """Compose the condition A + condition B superposition video."""
    import argparse
    global FFMPEG_EXE

    parser = argparse.ArgumentParser(
        description="Compose mixAB.mp4 (CondA + CondB superposition).")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help="Directory containing CondA.mp4 and "
                             "CondB.mp4 (default: data/videos).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for mixAB.mp4 "
                             "(default: results/videos).")
    parser.add_argument("--ffmpeg", type=str, default=FFMPEG_EXE,
                        help="Path to the ffmpeg executable.")
    args = parser.parse_args()

    FFMPEG_EXE = args.ffmpeg
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / "mixAB.mp4"

    print("Creating the CondA + CondB superposition...")

    filter_complex = build_filter_complex()
    run_ffmpeg([str(input_dir / "CondA.mp4"), str(input_dir / "CondB.mp4")],
               filter_complex, out_video)

    print(f"Video created: {out_video}")


if __name__ == "__main__":
    main()

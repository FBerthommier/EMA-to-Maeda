"""Superimpose the condition B articulation video on the reference
articulation video.

The reference video (CondRef.mp4, input 0) forms the background, and
the condition B video (CondB.mp4, input 1) is scaled, hue-shifted by
120 degrees (green to blue), converted to RGBA with 40% transparency,
and overlaid. The output is written to results/videos/mixBRef.mp4.

Inputs are looked up in data/videos/ relative to the repository root.

Original script: SynthsylShort/superpose2b.py
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
        str: The filter graph:
            1) scale input 0 to a height of 480 px at 25 fps;
            2) scale input 1, shift the hue by 120 degrees (green to
               blue), convert to RGBA with 40% alpha;
            3) overlay condition B on top of the reference.
    """
    return (
        # 1) CondA (input 0): simple resize
        "[0:v]fps=25,scale=-1:480[condA];"
        # 2) CondB (input 1): resize + green-to-blue hue shift (120 deg)
        #    + RGBA conversion for 40% transparency
        "[1:v]fps=25,scale=-1:480,hue=h=120,format=rgba,"
        "colorchannelmixer=aa=0.40[condB];"
        # 3) Overlay: reference in the background, CondB on top
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
    """Compose the condition B + reference superposition video."""
    import argparse
    global FFMPEG_EXE

    parser = argparse.ArgumentParser(
        description="Compose mixBRef.mp4 (CondRef + hue-shifted CondB at "
                    "40% alpha).")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help="Directory containing CondRef.mp4 and "
                             "CondB.mp4 (default: data/videos).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for mixBRef.mp4 "
                             "(default: results/videos).")
    parser.add_argument("--ffmpeg", type=str, default=FFMPEG_EXE,
                        help="Path to the ffmpeg executable.")
    args = parser.parse_args()

    FFMPEG_EXE = args.ffmpeg
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / "mixBRef.mp4"

    print("Creating the superposition (light blue tint, 40% alpha)...")

    filter_complex = build_filter_complex()
    # Input order: CondB.mp4 (unused by the filter graph but kept to
    # reproduce the original command line), then CondRef.mp4.
    run_ffmpeg([str(input_dir / "CondB.mp4"), str(input_dir / "CondRef.mp4")],
               filter_complex, out_video)

    print(f"Video created: {out_video}")


if __name__ == "__main__":
    main()

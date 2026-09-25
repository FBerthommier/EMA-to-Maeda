"""Batch preparation of the reference articulatory parameters
(copyparam).

For every utterance (X in 10..16, Y in 2..11), the 14-channel EMA data
stored in MATLAB .mat files (data/mat/VV_Y0X.mat) are reduced to 6
Maeda-like parameters, decimated, low-pass filtered, z-normalized, and
renormalized with the mean and standard deviation of the corresponding
Maeda reference model (P0X). The result is saved to
data/copyparam/CP_Y0X.npy.

Original script: SynthsylShort/batch_param.py
"""

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]


def reduce_to_6vars_mat(data14):
    """
    Reduce 14 EMA channels to 6 Maeda-like articulatory parameters.

    Args:
        data14 (np.ndarray): EMA data of shape (T, 14), columns ordered
            as TDx, TBx, TMx, TTx, ULx, LLx, JAWx,
            TDy, TBy, TMy, TTy, ULy, LLy, JAWy.

    Returns:
        np.ndarray: Parameters of shape (T, 6), in the same order as the
            Maeda model (Jaw, Body, Drsm, Tip, LipP, LipH).
    """
    td_x, tb_x, tm_x, tt_x, ul_x, ll_x, jaw_x, \
        td_y, tb_y, tm_y, tt_y, ul_y, ll_y, jaw_y = data14.T

    tip = tt_x + tt_y + tm_x + tm_y
    drsm = tb_x + td_x + tb_y + td_y
    body = tm_x + tb_x + td_x
    jaw = jaw_y
    lip_p = -(ul_x + ll_x)
    lip_h = ul_x + ll_x - ll_y + ul_y

    return np.vstack([jaw, body, drsm, tip, lip_p, lip_h]).T


def load_ema_reduce(mat_path):
    """
    Load one MATLAB EMA file and reduce it to 6 articulatory parameters.

    The X/Y coil channels are extracted with MATLAB-validated 0-based
    indices (X: 2-8, Y: 11-17).

    Args:
        mat_path (Path or str): Path to a VV_*.mat file containing a
            "data" array of shape (T, 27).

    Returns:
        np.ndarray: Parameters of shape (6, T).
    """
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    data = mat["data"]  # (T, 27)

    idx_x = [2, 3, 4, 5, 6, 7, 8]
    idx_y = [11, 12, 13, 14, 15, 16, 17]
    indices = idx_x + idx_y

    ema = data[:, indices].astype(np.float32)

    if ema.shape[1] != 14:
        raise ValueError("Incorrect extraction: 14 parameters expected")

    ema_reduce = reduce_to_6vars_mat(ema)

    return ema_reduce.T  # (6, T)


def zero_phase_lowpass_filter(data, cutoff_freq=8.0, sample_rate=100.0, order=4):
    """
    Apply a zero-phase (forward-backward) Butterworth low-pass filter.

    Args:
        data (np.ndarray): Signal, filtered along the last axis.
        cutoff_freq (float): Cutoff frequency in Hz.
        sample_rate (float): Sampling rate in Hz.
        order (int): Butterworth filter order.

    Returns:
        np.ndarray: Filtered signal, same shape as data.
    """
    b, a = signal.butter(order, cutoff_freq / (sample_rate / 2), btype="low")
    return signal.filtfilt(b, a, data, axis=-1)


def process_utterance(fname2, fname1, out_dir=None):
    """
    Process one utterance: load, decimate, filter, normalize, and save
    the copyparam file.

    Args:
        fname2 (str): Utterance code, e.g. "2012" for Y=2, X=12.
        fname1 (str): Reference model code, e.g. "P012".
        out_dir (Path or None): Output directory for CP_<fname2>.npy;
            None writes to the canonical ``data/copyparam/`` location.
    """
    # Reference model statistics (mean and std of the first 6 parameters)
    ema_data1_path = REPO_ROOT / "data" / "maeda" / f"{fname1}.npy"
    data = np.load(ema_data1_path, allow_pickle=True).item()
    ema_data = np.squeeze(data["Pv"]).T

    mean_data1 = np.mean(ema_data, axis=1, keepdims=True)[:6]
    std_data1 = np.std(ema_data, axis=1, keepdims=True)[:6]

    # Natural EMA signal, decimated, filtered, and z-normalized
    ema_data2_path = REPO_ROOT / "data" / "mat" / f"VV_{fname2}.mat"
    ema_data_brut = load_ema_reduce(ema_data2_path)

    ema_data_down = signal.decimate(ema_data_brut, 5, axis=1, n=4)
    ema_data = zero_phase_lowpass_filter(ema_data_down)

    ema_data2 = (ema_data - np.mean(ema_data, axis=1, keepdims=True)) / \
        (np.std(ema_data, axis=1, keepdims=True) + 1e-9)

    # Renormalize with the model statistics and save
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "copyparam"
    copy_synth_path = Path(out_dir) / f"CP_{fname2}.npy"
    copy_synth_path.parent.mkdir(parents=True, exist_ok=True)
    ema_data_synth = (ema_data2 * std_data1) + mean_data1
    np.save(copy_synth_path, ema_data_synth.T)


def main():
    """Run the batch copyparam generation over all utterances."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch copyparam generation from the raw EMA .mat "
                    "corpus (VV_*.mat -> CP_*.npy).")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Root whose copyparam/ subdirectory receives "
                             "the CP_*.npy outputs (default: data/copyparam).")
    args = parser.parse_args()

    out_dir = None if args.out_root is None else Path(args.out_root) / "copyparam"

    for x in range(10, 17):  # X from 10 to 16 (outer loop)
        # Reference model name
        fname1 = f"P0{x}"

        for y in range(2, 11):  # Y from 2 to 10 (inner loop)
            # Utterance name built with the Y0X rule
            fname2 = f"{y}0{x}"
            process_utterance(fname2, fname1, out_dir=out_dir)


if __name__ == "__main__":
    main()

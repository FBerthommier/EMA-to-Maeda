"""Canonical Time Warping (CTW) alignment and filtering utilities."""

import numpy as np
from scipy import signal
from tslearn.metrics import ctw_path


def apply_ctw_path(ref_ema, target_ema):
    """
    Align ref_ema onto target_ema with CTW.

    The CTW path is computed between ref_ema and target_ema, and the
    aligned signal is reconstructed on the time grid of target_ema
    (i.e. ref_ema is warped toward target_ema). For each target frame,
    the corresponding reference frames are averaged; gaps are filled by
    linear interpolation.

    Args:
        ref_ema (np.ndarray): Reference signal of shape (T_ref, D).
        target_ema (np.ndarray): Target signal of shape (T_tar, D).

    Returns:
        tuple:
            aligned (np.ndarray): Warped reference, shape (T_tar, D).
            path_matrix (np.ndarray): Warping path of shape (n_points, 2),
                each row being (idx_ref, idx_tar).
    """
    # CTW path
    result = ctw_path(ref_ema, target_ema)
    path = result[0] if isinstance(result, tuple) else result
    path = path.path_ if hasattr(path, "path_") else path
    path_list = list(path)

    # Split reference / target indices
    idx_ref, idx_tar = zip(*path_list)
    idx_ref = np.array(idx_ref)
    idx_tar = np.array(idx_tar)

    t_tar = target_ema.shape[0]
    n_dim = ref_ema.shape[1]

    # aligned = ref_ema warped onto the time grid of target_ema
    aligned = np.zeros((t_tar, n_dim))

    # For each target frame, average the corresponding reference frames
    for t in range(t_tar):
        indices = np.where(idx_tar == t)[0]
        if len(indices) > 0:
            aligned[t] = np.mean(ref_ema[idx_ref[indices]], axis=0)
        else:
            aligned[t] = np.nan  # gap, will be interpolated

    # Interpolate gaps in each dimension
    for d in range(n_dim):
        column = aligned[:, d]
        valid = ~np.isnan(column)
        if not valid.any():
            continue
        if valid.sum() < t_tar:
            aligned[:, d] = np.interp(
                np.arange(t_tar),
                np.where(valid)[0],
                column[valid],
            )

    # Warping path stored as (idx_ref, idx_tar) rows
    path_matrix = np.vstack([idx_ref, idx_tar]).T

    return aligned, path_matrix


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

"""Plotting utilities: parameter trajectories, formants, spectrograms.

Refactored from SynthsylShort/synthSYLVcourtes.py (plot_Pv, plot_formants,
spectreplot). The dead ``if 0:`` formant-overlay branch and its helpers
(lpc_formants / extract_formants) were removed.
"""

import matplotlib.pyplot as plt
import numpy as np

from synthesis.config import PARAM_LABELS


def close_on_enter(event):
    """Close the figure that emitted the key event (Enter key)."""
    if event.key == "enter":
        plt.close(event.canvas.figure)


def plot_Pv(Pv,
            labels=PARAM_LABELS,
            x_label="Frame index",
            y_label="Parameter value",
            title="Evolution of articulatory parameters (Pv)",
            x_range=None,
            y_range=None,
            grid=True,
            fig_number=None):
    """Plot the columns of Pv (articulatory parameters) versus time."""
    T, N = Pv.shape
    t = np.arange(T)

    fig = plt.figure(fig_number) if fig_number is not None else plt.figure()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)

    plt.plot(t, Pv)

    xmin, xmax = x_range if x_range is not None else (0, T - 1)
    ymin, ymax = y_range if y_range is not None else (-4, 5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.legend(labels, loc="best", ncol=N, frameon=False)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if grid:
        plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def plot_formants(fval,
                  labels=("F1", "F2", "F3"),
                  x_label="Frame index",
                  y_label="Formant frequency (Hz)",
                  title="Evolution of Formant Frequencies",
                  x_range=None,
                  y_min=0,
                  top_margin=0.2,
                  grid=True,
                  fig_number=None):
    """Plot formants F1, F2, F3 versus time."""
    fig = plt.figure(fig_number) if fig_number is not None else plt.figure()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)

    # Prepare fval as an (T, 3) array.
    arr = np.squeeze(fval)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"fval must be shape (T,3) or (3,T); got {arr.shape}")

    T = arr.shape[0]
    t = np.arange(T)

    plt.plot(t, arr)

    xmin, xmax = x_range if x_range is not None else (0, T - 1)
    plt.xlim(xmin, xmax)

    ymax_data = np.nanmax(arr)
    ymin = y_min
    ymax = ymax_data + (ymax_data - ymin) * top_margin
    plt.ylim(ymin, ymax)

    plt.legend(
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=True,
        borderpad=0.5,
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


def spectreplot(y, sr, nb, Rs, boolmel):
    """Display the spectrogram of a synthesized signal."""
    import librosa
    import librosa.display

    frame_length = 400  # 20 ms
    hop_length = 200    # 10 ms (50% overlap)

    if boolmel:
        n_mels = 64  # Number of Mel bands
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length,
                                           hop_length=hop_length, n_mels=n_mels)
    else:
        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    # Convert amplitude to dB.
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(nb * 3, 3))
    fig.canvas.mpl_connect("key_press_event", close_on_enter)

    if boolmel:
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="mel")
    else:
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                 x_axis="time", y_axis="linear")

    plt.colorbar(format="%+2.0f dB")
    plt.title(Rs)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 4500)

    plt.tight_layout()
    plt.show(block=True)

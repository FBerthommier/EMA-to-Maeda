"""Audio playback and output-file writing utilities.

Refactored from SynthsylShort/synthSYLVcourtes.py (the ``play`` function and
the wav/npy writing lines of main()).
"""

import numpy as np
from scipy.io.wavfile import write

from synthesis import config


def play(signal):
    """Play a 20 kHz signal via sounddevice (non-blocking).

    Playback is skipped unless config.PLAYBACK_ENABLED is True; this does not
    affect the synthesized data.
    """
    if config.PLAYBACK_ENABLED:
        import sounddevice as sd
        sd.play(signal, config.FS)


def save_wav(sig, path, fs=config.FS, scaling=1.20):
    """Normalize and write a float signal as int16 wav (as in the original)."""
    signal_int16 = np.int16(32767 * sig / (scaling * np.max(np.abs(sig))))
    write(str(path), fs, signal_int16)
    return path


def save_parameters(params, path):
    """Save the synthesis parameters dict (fval, Pv, dur) as an .npy file."""
    np.save(str(path), params)
    return path

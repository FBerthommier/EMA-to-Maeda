"""LPC source-filter audio synthesis from spectral frames.

Refactored from SynthsylShort/synthSYLVcourtes.py (source generation by
L. Girin, GIPSA-lab, 2009; envelope shaping and LPC filtering).

Converts a matrix of magnitude spectra (one row per 10 ms frame) into a
time signal: each spectral frame is converted to an LPC filter, a pulse
train with F0 jitter excites the time-varying lattice filter, and the
result is shaped by a syllable-type energy envelope.
"""

import numpy as np
from numpy.polynomial import Polynomial  # noqa: F401  (kept for API parity)
from scipy.interpolate import interp1d
from scipy.signal import lfilter

from synthesis import config  # noqa: F401  (kept for API parity)
from synthesis.acoustic_model import Hfreq2lpc, poly2rc
from synthesis.audio_io import play


def gen_src_3(L, fs, F0, flag_stretch):
    """Generate a glottal-pulse source signal.

    Parameters
    ----------
    L : int
        Length of the generated signal (samples).
    fs : int
        Sampling frequency (Hz).
    F0 : ndarray
        Fundamental frequency trajectory (Hz).
    flag_stretch : int
        0: fixed-shape pulses convolved with an impulse train;
        otherwise: period-adapted pulses placed at each F0 period.
    """
    M = len(F0)
    if M != L:
        F0 = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, M), F0)

    if flag_stretch == 0:
        F0_max = np.max(F0)
        T = int(fs / F0_max / 0.8)
        Tp = int(T / 3)
        Tn = int(T / 4)
        t1 = np.arange(1, Tp + 1)
        t2 = np.arange(Tp + 1, Tp + Tn + 1)
        pulse = np.concatenate([(3 * (t1 / Tp) ** 2 - 2 * (t1 / Tp) ** 3),
                                (1 - ((t2 - Tp) / Tn) ** 2)])
        train = np.zeros(L)
        ind = 0
        while True:
            train[ind] = 1
            ind += int(fs / F0[ind])
            if ind >= L:
                break
        s = np.convolve(train, pulse, mode="full")
    else:
        s = np.zeros(L)
        ind = 0
        while True:
            T = int(fs / F0[ind])
            Tp = int(T / 3)
            Tn = int(T / 6)
            t1 = np.arange(1, Tp + 1)
            t2 = np.arange(Tp + 1, Tp + Tn + 1)
            pulse = np.concatenate([(3 * (t1 / Tp) ** 2 - 2 * (t1 / Tp) ** 3),
                                    (1 - ((t2 - Tp) / Tn) ** 2)])
            s[ind:ind + Tp + Tn] = pulse
            ind += int(fs / F0[ind])
            if ind >= L:
                break

    return s[:L]


def poly2rc_to_signal_filter(a):
    """Return the reflection coefficients of LPC polynomial ``a``."""
    return poly2rc(a)


def f_lpc_exc2sig(e, step_size, trans_size, fs, LPC):
    """Filter an excitation signal through time-varying LPC lattice filters.

    Parameters
    ----------
    e : ndarray
        Excitation signal.
    step_size : float
        Hop size between successive filters (s).
    trans_size : float
        Duration of the inter-frame filter transition (s).
    fs : int
        Sampling frequency (Hz).
    LPC : ndarray
        Matrix of LPC coefficient vectors (one per row).
    """
    N_step = int(step_size * fs)
    N_trans = int(trans_size * fs)

    L = len(e)
    M = L // N_step
    if M != LPC.shape[0]:
        print("\nProblem with the size of the data\n")
        return 0

    p = LPC.shape[1] - 1
    sig = np.zeros(L)
    e_f = np.zeros(p + 1)
    e_b = np.zeros(p + 1)

    a = LPC[0, :]
    k_P = poly2rc_to_signal_filter(a)

    for m in range(M):
        a1 = LPC[m, :]
        k_C = poly2rc_to_signal_filter(a1)

        dk = (k_C - k_P) / N_trans

        for ind in range(N_step):
            if ind < N_trans:
                k_P += dk

            e_f[p] = e[ind + m * N_step]

            for ind_p in range(p):
                e_f[p - ind_p - 1] = e_f[p - ind_p] - k_P[p - ind_p - 1] * e_b[p - ind_p - 1]
                e_b[p - ind_p] = e_b[p - ind_p - 1] + k_P[p - ind_p - 1] * e_f[p - ind_p - 1]

            e_b[0] = e_f[0]
            sig[ind + m * N_step] = e_f[0]

    return sig


def default_f0(N_t):
    """Default F0 contour (cubic interpolation of 100/130/110/90 Hz) + jitter."""
    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
                  np.array([100, 130, 110, 90]),
                  kind="cubic")(np.arange(1, N_t + 1))

    np.random.seed(0)
    F0 = F0 + np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5
    return F0


def synthfen(word1, cexp, cdec, durfen):
    """Energy envelope for a multi-syllable word, frame by frame.

    ``word1`` is a list of syllable-type tokens (e.g. ['O', 'CV', 'VC', 'F']).
    Pairs of consecutive tokens determine the envelope segments ('CV' onset,
    'VC' coda, 'CC' cluster, 'OC' initial silence, ...).
    """
    word = "".join(word1)
    nbfen = len(word) - 1
    fen3 = np.hanning(2 * durfen)
    fen2 = np.array([])

    for k in range(nbfen):
        syl = word[k:k + 2]
        if syl == "OC":
            fen2 = np.concatenate([fen2, np.zeros(durfen)])
        elif syl == "OV":
            pass
        elif syl == "CF":
            fen2 = np.concatenate([fen2, fen3[:durfen] ** cexp])
        elif syl == "VF":
            pass  # Not implemented (as in the original script)
        elif syl == "VV":
            fen2 = np.concatenate([fen2, np.ones(2 * durfen)])
        elif syl in ["CV", "cV"]:
            fen2 = np.concatenate([fen2, fen3[:durfen] ** cexp])
        elif syl in ["VC", "Vc"]:
            fen2 = np.concatenate([fen2, fen3[durfen:] ** cdec])
        elif syl == "CC":
            fen2 = np.concatenate([
                fen2,
                0.5 * fen3[:durfen] ** cexp,
                0.5 * fen3[durfen:] ** (3 * cdec),
            ])
        elif syl in ["Cc", "cC"]:
            product = (fen3[:durfen] ** cexp) * (fen3[durfen:] ** (1 * cdec))
            fen2 = np.concatenate([fen2, product])

    return fen2


def synthsimpleSYL(matspec, syl, cf0, cexp, cdec):
    """Synthesize one syllable audio signal from spectral frames.

    Parameters
    ----------
    matspec : ndarray
        Magnitude spectra, one row per frame.
    syl : str
        Syllable type ('CV', 'VCV', 'CCV', ...), selects the envelope.
    cf0, cexp, cdec : coefficients (kept for API parity with the original).
    """
    fs = 20000  # 20 kHz, as in the original script
    lg, _ = matspec.shape

    M = lg
    p = 30

    T_s = 2 * 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))

    A = []
    for k in range(lg):
        G, Ai = Hfreq2lpc(matspec[k, :], p)
        A.append(Ai.flatten())

    F0 = default_f0(N_t)
    e = gen_src_3(N_t, fs, F0, 0)
    sig = f_lpc_exc2sig(e, T_s, 1e-3, fs, np.array(A))

    lg = len(sig)
    lg2 = round(lg / 4)
    han = np.hanning(lg2)
    handeb = han[:round(lg2 / 2)]
    hanfin = han[round(lg2 / 2):]
    fen = np.concatenate((handeb, np.ones(lg - lg2), hanfin ** 2))

    if syl == "CV":
        fen3 = np.hanning(lg)
        fen2 = np.concatenate((np.zeros(round(lg / 2)), fen3[:round(lg / 2)] ** cexp))
    elif syl == "CVV":
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.zeros(round(lg / 4)), fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 2))))
    elif syl == "VCVV":
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp, np.ones(round(lg / 2))))
    elif syl == "VC":
        fen3 = np.hanning(lg)
        fen2 = np.concatenate((fen3[round(lg / 2):] ** cdec, fen3[:round(lg / 2)] ** cexp))
    elif syl == "VVC":
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.ones(round(lg / 2)), fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == "VCV":
        fen3 = np.hanning(lg)
        fen2 = np.concatenate((fen3[round(lg / 2):] ** cdec, fen3[:round(lg / 2)] ** cexp))
    elif syl == "CVC":
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((np.zeros(round(lg / 4)), fen3[:round(lg / 4)] ** cexp, fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == "VCVC":
        fen3 = np.hanning(round(lg / 2))
        fen2 = np.concatenate((fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp, fen3[round(lg / 4):] ** cdec, fen3[:round(lg / 4)] ** cexp))
    elif syl == "CVVC":
        fen3 = np.hanning(round(lg / 3))
        fen2 = np.concatenate((np.zeros(round(lg / 6)), fen3[:round(lg / 6)] ** cexp, np.ones(round(lg / 3)), fen3[round(lg / 6):] ** cdec, fen3[:round(lg / 6)] ** cexp))
    elif syl == "VCVVC":
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp))
    elif syl == "CCV":
        fen3 = np.hanning(round(lg / 1.5))
        fen2 = np.concatenate((np.zeros(round(lg / 3)), (fen3[:round(lg / 3)] ** cexp) * fen3[round(lg / 3):] ** cdec, fen3[:round(lg / 3)] ** cexp))
    elif syl == "CCVV":
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.zeros(round(lg / 5)), (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5)), np.ones(round(lg / 5))))
    elif syl == "VCCVV":
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((fen3[round(lg / 5):] ** cdec, (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp, np.ones(round(lg / 5)), np.ones(round(lg / 5))))
    elif syl == "VCC":
        fen3 = np.hanning(round(lg / 1.5))
        fen2 = np.concatenate((fen3[round(lg / 3):] ** cdec, (fen3[:round(lg / 3)] ** cexp) * fen3[round(lg / 3):] ** cdec, fen3[:round(lg / 3)] ** cexp))
    elif syl == "VVCC":
        fen3 = np.hanning(round(lg / 2.5))
        fen2 = np.concatenate((np.ones(round(lg / 5)), np.ones(round(lg / 5)), fen3[round(lg / 5):] ** cdec, (fen3[:round(lg / 5)] ** cexp) * fen3[round(lg / 5):] ** cdec, fen3[:round(lg / 5)] ** cexp))
    elif syl == "VCCV":
        fen3 = np.hanning(round(lg / 1.5))
        fen2 = np.concatenate((fen3[round(lg / 3):] ** cdec, (fen3[:round(lg / 3)] ** cexp) * fen3[round(lg / 3):] ** cdec, fen3[:round(lg / 3)] ** cexp))

    sig = lfilter([1, -0.9375], 1, fen * fen2 * sig)
    sig = sig / (1.01 * np.max(np.abs(sig)))
    play(sig)
    return sig


def synthsimpleWORDfen(matspec, word, cf0, cexp, cdec, dur):
    """Synthesize a full multi-syllable word with a per-syllable envelope."""
    fs = 20000

    lg, _ = matspec.shape
    M = lg
    p = 30

    T_s = 2 * 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))

    A = []
    for k in range(lg):
        G, Ai = Hfreq2lpc(matspec[k, :], p)
        A.append(Ai.flatten())

    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
                  np.array([100, 130, 110, 90]),
                  kind="cubic")(np.arange(1, N_t + 1))

    np.random.seed(0)
    F0 += np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5  # jitter
    e = gen_src_3(N_t, fs, F0, 0)
    sig = f_lpc_exc2sig(e, T_s, 1e-3, fs, np.array(A))

    lg = len(sig)
    lg2 = round(lg / 4)
    han = np.hanning(lg2)
    handeb = han[:round(lg2 / 2)]
    hanfin = han[round(lg2 / 2):]
    # Softened release (dampening removed relative to an earlier version).
    fen = np.concatenate([handeb, np.ones(lg - lg2), hanfin ** 0.5])
    fen2 = synthfen(word, cexp, cdec, int(dur * fs * T_s))
    sig = lfilter([1, -0.9375], 1, fen * fen2 * sig)

    sig = sig / (1.01 * np.max(np.abs(sig)))

    play(sig)
    return sig

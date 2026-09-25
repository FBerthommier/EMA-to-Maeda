"""Shared helpers for the VLAM audio/video stimulus generation scripts.

This module contains everything duplicated across the legacy scripts of
``SynthsylShort/``:

- Path helpers (repo-root ``data/`` and ``results/videos/`` directories).
- The VLAM articulatory model (sagittal contour and area-function forward
  model, ``vlam2009NN``).
- The acoustic chain used to compute vocal-tract transfer functions and
  formants from area functions, and to synthesize speech with a
  time-varying LPC lattice filter driven by a glottal pulse train.
- Envelope extraction from real recordings (Hilbert envelope, 8 Hz
  low-pass, resampling).
- Articulatory parameter file readers / resamplers.
- Matplotlib frame rendering and OpenCV AVI writing, plus an ffmpeg
  helper to mux the audio track into the final MP4.

The MP4 videos produced with these tools correspond to the article
videos: ``CondA.mp4`` (condition A, reference parameters),
``CondB.mp4`` (condition B, CTW-random converted parameters) and
``mixAB.mp4`` (side-by-side mix of both conditions).

Refactored from ``SynthsylShort/VLAMaudmaker.py``,
``SynthsylShort/VLAMaudmaker2.py`` and ``SynthsylShort/VLAMvidmaker2.py``.

External tools: ffmpeg must be available (used for audio muxing).
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, hilbert, lfilter, resample

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# Repo root is the parent of the video/ package directory (Github/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "videos"

# External binary used to mux audio into the MP4 containers.
FFMPEG = "ffmpeg"


# ---------------------------------------------------------------------------
# Small math utilities
# ---------------------------------------------------------------------------
def softrect(x, S):
    """Soft-rectification used to keep area values non-negative."""
    return (np.sqrt(x**2 + S) + x) / 2


def poly2rc(a):
    """Convert LPC prediction coefficients ``a`` to reflection coefficients.

    Uses the standard Levinson backward recursion. The first element of
    ``a`` must be 1.
    """
    if a[0] != 1:
        raise ValueError("The first coefficient of a must be 1.")

    p = len(a) - 1
    k = np.zeros(p)
    a_current = np.copy(a[1:])

    for i in range(p - 1, -1, -1):
        k[i] = a_current[i]
        if i > 0:
            flipped_a = np.flip(a_current[:i])
            a_current[:i] = (a_current[:i] - k[i] * flipped_a) / (1 - k[i] ** 2)

    return np.float64(k)


def levinson_durbin(r, p):
    """Levinson-Durbin recursion with MATLAB-compatible output.

    Parameters
    ----------
    r : ndarray
        Autocorrelation coefficients.
    p : int
        LPC model order.

    Returns
    -------
    a : ndarray
        LPC coefficients, with ``a[0] == 1``.
    e : float
        Residual squared error.
    """
    a = np.float64(np.zeros(p + 1))
    e = r[0]

    if e == 0:
        a[0] = 1
        return a, e

    a[0] = 1

    for i in range(1, p + 1):
        if i == 1:
            k = r[1] / e
        else:
            k = (r[i] - np.dot(a[1:i], r[i - 1:0:-1])) / e

        a_new = a[1:i] - k * np.flip(a[1:i])
        a[i] = k
        a[1:i] = a_new

        e *= (1 - k**2)
        if e < 0:
            e = 0

    a[1:] = -a[1:]
    return np.float64(a), e


# ---------------------------------------------------------------------------
# LPC synthesis (source-filter)
# ---------------------------------------------------------------------------
def f_lpc_exc2sig(e, step_size, fs, LPC):
    """Synthesize a signal from an excitation and time-varying LPC filters.

    Interpolates the reflection coefficients linearly over a 1 ms
    transition window between successive analysis frames (10 ms hop).
    The lattice structure follows the classic cross-branch formulation
    (Girin, GIPSA-lab).
    """
    N_step = int(step_size * fs)
    N_trans = int(1e-3 * fs)

    L = len(e)
    M = L // N_step
    if M != LPC.shape[0]:
        print("\nProblem with the size of the data\n")
        return 0

    p = LPC.shape[1] - 1
    sig = np.zeros(L)
    e_f = np.zeros(p + 1)
    e_b = np.zeros(p + 1)

    k_P = poly2rc(LPC[0, :])

    for m in range(M):
        k_C = poly2rc(LPC[m, :])
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


def gen_src_3(L, fs, F0, flag_stretch):
    """Generate a glottal-like pulse train following an F0 trajectory.

    Each pulse is a rising parabola followed by a falling parabola
    (source generation v2.0, L. Girin, GIPSA-lab, 2009).
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
        s = np.convolve(train, pulse, mode='full')
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


def Hfreq2lpc(H, p):
    """Convert a frequency-response sampling into an LPC model.

    Parameters
    ----------
    H : ndarray
        Spectrum samples (linear scale, positive frequencies).
    p : int
        LPC order.

    Returns
    -------
    g : float
        LPC gain.
    a : ndarray
        LPC coefficients.
    """
    H = np.asarray(H).flatten()
    N = len(H)

    # Complete with the "negative frequencies".
    Hr = np.concatenate([H, [H[N - 1]], np.flipud(H[1:N - 1])])

    # Autocorrelation coefficients = IFFT of the power spectrum.
    R = np.float64(np.real(np.fft.ifft(np.abs(Hr) ** 2)))
    R = R[:p + 1]

    a, e = levinson_durbin(R, p)
    g = np.sqrt(e)

    return g, np.float64(a)


# ---------------------------------------------------------------------------
# Vocal-tract acoustics (area function -> transfer function / formants)
# ---------------------------------------------------------------------------
# Physical constants of the vocal-tract acoustic model:
# c [cm/s], air density [g/cm^3], ..., wall parameters.
CONST_DAT = np.array([35100, 1.14e-3, 5.5e-5, 1.4, 1.86e-4, 0.24, 1600, 1.4])


def spectrelec(w, A, zr, l, no_vibration):
    """Transmission-line (electrical analogy) transfer function of a tube chain."""
    c = CONST_DAT[0]
    ro = CONST_DAT[1]
    lambda_ = CONST_DAT[2]
    eta = CONST_DAT[3]
    mu = CONST_DAT[4]
    cp = CONST_DAT[5]
    bp = CONST_DAT[6]
    mp = CONST_DAT[7]

    S = 2 * np.sqrt(A * np.pi)
    L = ro / A * l
    C = A * l / (ro * c * c)

    R_coef = np.sqrt(ro * mu / (2 * w))
    G_coef = (eta - 1) / (ro * c**2) * np.sqrt(lambda_ * w / (2 * cp * ro))
    R = S * l / (A**2) * R_coef
    G = S * l * G_coef

    YP_coef = 1 / (bp**2 + mp**2 * w**2)
    YP = S * l * ((bp - 1j * mp * w) * YP_coef)

    if no_vibration == 1:
        YP = 0

    Z = R + 1j * L * w
    Y = G + 1j * C * w + YP
    aa = (1 + (Z * Y / 2))

    bb = -(Z + Z**2 * Y / 4)
    cc = -Y
    dd = aa
    aaa = aa[0, :]
    bbb = bb[0, :]
    ccc = cc[0, :]
    ddd = dd[0, :]

    for ind in range(len(A) - 1):
        proda = aa[ind + 1, :] * aaa + bb[ind + 1, :] * ccc
        prodb = aa[ind + 1, :] * bbb + bb[ind + 1, :] * ddd
        prodc = cc[ind + 1, :] * aaa + dd[ind + 1, :] * ccc
        prodd = cc[ind + 1, :] * bbb + dd[ind + 1, :] * ddd
        aaa = proda
        bbb = prodb
        ccc = prodc
        ddd = prodd
        H = np.ones_like(aaa) / (aaa - ccc * zr)
    return H


def aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration):
    """Transfer function of the oral tract from its area function."""
    c = CONST_DAT[0]
    rho = CONST_DAT[1]
    f = np.linspace(Fmin, Fmax, nbfreq)
    w = np.squeeze(2 * np.pi * f)

    Zr_oral = (rho / (2 * np.pi * c) * (w ** 2)
               + 1j * 8 * rho / (3 * np.pi * np.sqrt(np.pi * area[-1, 1])) * w)
    H_oral = spectrelec(w, area[:, 1].reshape(-1, 1), Zr_oral,
                        area[:, 0].reshape(-1, 1), no_vibration)
    return H_oral


def aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration):
    """Oral-tract transfer function with already-found formants cancelled.

    Used iteratively so that each Newton search finds the next formant.
    """
    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration)

    f = np.linspace(Fmin, Fmax, nbfreq)
    if np.isrealobj(f):
        SI = 1j * 2 * np.pi * f
    else:
        SI = 2 * np.pi * 1j * f

    for I in range(NF):
        SI1 = complex(F_form[I, 2] * np.pi, F_form[I, 1] * 2 * np.pi)
        H_eq *= np.squeeze(((SI - SI1) * (SI - np.conj(SI1))) / (SI1 * np.conj(SI1)))

    return H_eq


def nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration):
    """Newton-Raphson search of one formant (pole) of the oral transfer function.

    Legacy Fortran routine (nraph3, Sanchez/Badin, ICP Grenoble) ported to
    Python: searches for a complex root s = -BP*pi + j*FE*2*pi.
    """
    DELTAS = complex(30, 30)
    SEUIL = 0.3
    ITER = 1
    SI = complex(-BNPE * np.pi, FE.item() * 2 * np.pi)

    F = np.nan
    BP = np.nan
    while ITER < ITERMX:
        FIcx = -1j * SI / (2 * np.pi)
        Q = 1 / aire2spectre_cor_oral(area, 1, FIcx, FIcx, F_form, NF, no_vibration)
        SIP = SI + DELTAS
        FIPcx = -1j * SIP / (2 * np.pi)
        QP = 1 / aire2spectre_cor_oral(area, 1, FIPcx, FIPcx, F_form, NF, no_vibration)
        Q1D = (QP - Q) / DELTAS
        SISU = SI - (Q / Q1D)
        if abs(SISU - SI) < SEUIL:
            F = np.imag(SISU) / (2 * np.pi)
            BP = -np.real(SISU) / np.pi
            return F, BP
        SI = SISU
        ITER += 1

    return np.float64(F), np.float64(BP)


def vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, no_vibration):
    """Extract formants from a vocal-tract area function.

    Iteratively finds the poles of the oral transfer function up to
    ``Fmax`` and returns the transfer function plus the first three
    formant frequencies.
    """
    f = np.linspace(Fmin, Fmax, nbfreq)

    FE = 150
    BNPE = 50
    FINC = 100
    ITERMX = 100
    FMAX = Fmax

    F_form = np.zeros((100, 3))

    NF = 0
    F = 0
    while F <= FMAX:
        if NF > 0:
            FE = F + FINC
        H_eq = aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration)
        ind_maxi, _ = find_peaks(np.asarray(20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        ind_mini, _ = find_peaks(np.asarray(-20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        frq_max = f[ind_maxi]
        FEST = frq_max
        if any(FEST):
            FE = FEST[0]

        F, BP = nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration)
        F_form[NF, :] = [0, F.item(), BP.item()]
        NF += 1

    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration)

    nbformants = 3
    F_form1 = np.zeros((nbformants, 1))
    for k in range(min(nbformants, NF)):
        F_form1[k] = F_form[k, 1]

    return H_eq, F_form1


# ---------------------------------------------------------------------------
# VLAM articulatory model (sagittal contour + area function)
# ---------------------------------------------------------------------------
def init_vlam_length(Ap, L):
    """Initialize the VLAM state dictionary for a vocal-tract length ``L``."""
    gui = {}
    gui['sagittal'] = np.zeros((58, 2))
    gui['inci'] = np.zeros((1, 2))
    gui['area'] = np.zeros((29, 2))
    gui['PD'] = 0
    gui['LH'] = 0
    gui['LHI'] = 0
    gui['B'] = 0
    gui['A'] = 0

    kage = 5.4749 * L - 407.1374
    gui['prm'] = np.zeros(11)
    gui['prm'][0:7] = Ap
    gui['prm'][7] = kage
    gui['prm'][8] = 0
    gui['prm'][10] = 0
    gui['prm'][9] = 0
    return gui


def vlam2009_nn(gui):
    """VLAM 2009 forward model: articulatory parameters -> sagittal contour and areas.

    Updates ``gui`` in place (sagittal outline, jaw/lip metrics and the
    29-section area function) from the 11-dimensional parameter vector
    ``gui['prm']`` (Jaw, Body, Dorsum, Apex, LipH, LipP, Larynx, Age,
    plus flattening/rotation extras).
    """
    # Extract and reorder the parameters.
    parametres = np.zeros(10)
    parametres[0] = gui['prm'][0]  # Jaw
    parametres[1] = gui['prm'][1]  # Body
    parametres[2] = gui['prm'][2]  # Dorsum
    parametres[3] = gui['prm'][3]  # Apex
    parametres[4] = gui['prm'][5]  # Lip Protrusion
    parametres[5] = gui['prm'][4]  # Lip Height
    parametres[6] = gui['prm'][6]  # Larynx
    parametres[7] = gui['prm'][7]  # Age

    # Human mode scaling factors.
    parametres[8] = 0.8   # k_Pharynx
    parametres[9] = 0.35  # k_Mouth

    # Flattening and rotation parameters.
    Flat_P = -gui['prm'][8]
    Flat_T = -gui['prm'][10]
    phi = gui['prm'][9]
    AF_correc = 1

    # Constant shape matrices and offsets (tongue, lips, larynx, wall).
    A_tng = np.array([[1.000000, 0.000000, 0.000000, 0.000000],
                      [-0.464047, 0.098776, -0.251690, 0.000000],
                      [-0.328015, 0.337579, -0.283667, 0.000000],
                      [-0.213039, 0.485565, -0.283533, 0.000000],
                      [-0.302565, 0.705432, -0.379044, 0.000000],
                      [-0.327806, 0.786897, -0.388116, 0.000000],
                      [-0.325065, 0.852409, -0.285125, 0.000000],
                      [-0.325739, 0.904725, -0.142602, 0.000000],
                      [-0.313741, 0.926339, 0.021042, 0.000000],
                      [-0.288138, 0.924019, 0.131949, 0.000000],
                      [-0.249008, 0.909585, 0.250320, 0.000000],
                      [-0.196936, 0.882236, 0.369083, 0.000000],
                      [-0.128884, 0.830243, 0.499894, 0.000000],
                      [-0.040825, 0.730520, 0.651662, 0.112048],
                      [0.073420, 0.543080, 0.807947, 0.126204],
                      [0.202726, 0.230555, 0.919065, 0.163735],
                      [0.298853, -0.162541, 0.899074, 0.213884],
                      [0.332785, -0.491647, 0.748869, 0.243163],
                      [0.349955, -0.681313, 0.567615, 0.245295],
                      [0.377277, -0.771200, 0.410502, 0.249425],
                      [0.422713, -0.804874, 0.270513, 0.274015],
                      [0.474635, -0.797704, 0.129324, 0.314454],
                      [0.526087, -0.746938, -0.026201, 0.366149],
                      [0.549466, -0.643572, -0.190005, 0.422848],
                      [0.494200, -0.504012, -0.350434, 0.488056],
                      [0.448797, -0.417352, -0.445410, 0.500909]])

    s_tng = np.array([27.674635, 29.947931, 44.694466, 99.310226, 96.871323,
                      84.140404, 78.357513, 73.387718, 72.926758, 71.453232,
                      69.288765, 66.615509, 63.603722, 59.964859, 56.695446,
                      56.415058, 62.016468, 73.235176, 84.008438, 91.488312,
                      94.124176, 95.246323, 93.516365, 93.000343, 100.934669,
                      106.512482])

    u_tng = np.array([104.271675, 443.988434, 450.481689, 399.942200, 348.603088,
                      351.181122, 365.404633, 370.290955, 356.202301, 341.890167,
                      332.117523, 326.826599, 326.512512, 331.631989, 343.175323,
                      361.265900, 385.231201, 411.826599, 435.691711, 455.040466,
                      462.736023, 453.025055, 432.250488, 407.358368, 384.551056,
                      363.836212])

    A_lip = np.array([[1.000000, 0.000000, 0.000000],
                      [0.178244, -0.395733, 0.888897],
                      [-0.154638, 0.987971, 0.000000],
                      [-0.217332, 0.825187, -0.303429]])

    s_lip = np.array([27.674635, 33.068081, 99.392258, 213.996170])
    u_lip = np.array([104.271675, 122.812141, 135.938339, 460.440857])

    A_lrx = np.array([[1.000000, 0.000000],
                      [-0.208338, 0.262446],
                      [0.127814, 0.991798],
                      [-0.131840, 0.300784],
                      [0.097688, 0.934267]])

    s_lrx = np.array([27.674635, 41.593315, 65.562340, 44.372742, 66.147499])
    u_lrx = np.array([104.271675, 143.138733, -948.229309, 404.678223, -962.936401])

    u_wal = np.array([550.196533, 604.878601, 674.127197, 678.776489, 665.905579,
                      653.312134, 643.223511, 633.836243, 636.994202, 668.834290,
                      703.098267, 600, 610, 610, 605, 600, 600, 600, 600, 600,
                      600, 600, 600, 600, 600])

    # Initial scaling computations.
    k_age = parametres[7]
    k_age_max = 650
    Pharynx_scale = k_age * parametres[8] / k_age_max + 0.3
    Mouth_scale = k_age * parametres[9] / k_age_max + 0.65

    # Conversion factor between virtual and physical units.
    vp_map = 1 / 29.5
    TEKvt = 188.679245
    TEKvt *= vp_map

    # Rescale constants.
    s_tng /= TEKvt
    u_tng /= TEKvt
    s_lip /= TEKvt
    u_lip /= TEKvt
    s_lrx /= TEKvt
    u_lrx /= TEKvt
    u_wal /= TEKvt

    # Compute the midsagittal contours.
    v_tng = s_tng * (A_tng @ parametres[:4]) + u_tng
    v_lip = s_lip * (A_lip @ [parametres[0], parametres[4], parametres[5]]) + u_lip
    v_lip[v_lip < 0] = 0
    v_lrx = s_lrx * (A_lrx @ [parametres[0], parametres[6]]) + u_lrx

    # Reference points.
    ix0, iy0 = 2200, 2000
    r, dl, m1, m2, m3 = 5, 0.5, 14, 11, 6
    omega, theta = -11.25, 11.25

    # Convert to virtual units.
    r_vp = r / vp_map
    dlPharynx_vp = Pharynx_scale * dl / vp_map
    dlPalatal_vp = Mouth_scale * dl / vp_map
    ome = np.radians(omega)
    the = np.radians(theta)

    # Build the grid: pharyngeal part.
    dx_i = dlPharynx_vp * np.cos(ome - np.pi / 2)
    dy_i = dlPharynx_vp * np.sin(ome - np.pi / 2)
    dx_e = r_vp * np.cos(ome)
    dy_e = r_vp * np.sin(ome)

    igd1 = np.column_stack((dx_i * np.arange(m1 - 1, -1, -1) + ix0,
                            dy_i * np.arange(m1 - 1, -1, -1) + iy0))
    egd1 = np.column_stack((dx_e + igd1[:, 0], dy_e + igd1[:, 1]))

    # Palatal part.
    gam = the * np.arange(1, m2 + 1) + ome
    igd2 = np.column_stack((np.full(m2, ix0), np.full(m2, iy0)))
    egd2 = np.column_stack((r_vp * np.cos(gam) + ix0, r_vp * np.sin(gam) + iy0))

    # Labial part.
    dx_i = dlPalatal_vp * np.cos(gam[-1] + np.pi / 2)
    dy_i = dlPalatal_vp * np.sin(gam[-1] + np.pi / 2)
    dx_e = r_vp * np.cos(gam[-1])
    dy_e = r_vp * np.sin(gam[-1])

    igd3 = np.column_stack((dx_i * np.arange(1, m3 + 1) + ix0,
                            dy_i * np.arange(1, m3 + 1) + iy0))
    egd3 = np.column_stack((dx_e + igd3[:, 0], dy_e + igd3[:, 1]))

    # Final assembly.
    igd = np.vstack((igd1, igd2, igd3))
    egd = np.vstack((egd1, egd2, egd3))

    # Normalized direction vectors.
    p, q = egd[:, 0] - igd[:, 0], egd[:, 1] - igd[:, 1]
    s = np.hypot(p, q)
    vtos = np.column_stack((p / s, q / s))

    # Jaw computations.
    omg = np.radians(omega)
    JAW = 0

    # Point 1 (ivt1).
    x1, y1 = v_lrx[JAW + 1], v_lrx[JAW + 2]
    b = y1 - np.tan(omg + np.pi / 2) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2))
    y0 = np.tan(omg) * x0

    x1 = Pharynx_scale * (x1 - x0) + x0
    y1 = Pharynx_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2) * x0

    ivt1 = np.array([[Pharynx_scale * (x1 - x0) + x0 + ix0,
                      Pharynx_scale * (y1 - y0) + y0 + iy0]])

    # Point 2 (evt1).
    x1, y1 = v_lrx[JAW + 3], v_lrx[JAW + 4]
    b = y1 - np.tan(omg + np.pi / 2) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2))
    y0 = np.tan(omg) * x0

    x1 = Pharynx_scale * (x1 - x0) + x0
    y1 = Pharynx_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2) * x0

    evt1 = np.array([[Pharynx_scale * (x1 - x0) + x0 + ix0,
                      Pharynx_scale * (y1 - y0) + y0 + iy0]])

    # Scaling factors along the tract.
    scale_factor1 = Pharynx_scale * np.ones((8, 1))
    scale_factor2 = (Mouth_scale - Pharynx_scale) * np.arange(1, m2 + 1).reshape(-1, 1) / m2 + Pharynx_scale
    scale_factor3 = Mouth_scale * np.ones((6, 1))
    scale_factor = np.vstack((scale_factor1, scale_factor2, scale_factor3)).flatten()

    # Tongue contour.
    v = scale_factor[:25] * np.minimum(v_tng[1:26], u_wal)
    xy1 = igd[6:31] + vtos[6:31] * v.reshape(-1, 1)
    xy2 = igd[6:31] + vtos[6:31] * (u_wal * scale_factor[:25]).reshape(-1, 1)

    # Point assembly.
    ivt2 = (ivt1 + xy1[0]) / 2
    evt2 = (evt1 + xy2[0]) / 2
    ivt4 = np.vstack((ivt1, ivt2, xy1))
    evt4 = np.vstack((evt1, evt2, xy2))

    # Lip points.
    inci_x, inci_y, inci_lip = 2212.354492, 1999.574219, 0.8
    inci_x = (inci_x - 3000) / TEKvt
    inci_y = (inci_y - 1850) / TEKvt
    inci_lip_vp = inci_lip / vp_map

    the_rad = np.radians(theta)
    x1, y1 = inci_x, inci_y + inci_lip_vp
    b = y1 - np.tan(the_rad + np.pi / 2) * x1
    x0 = b / (np.tan(the_rad) - np.tan(the_rad + np.pi / 2))
    y0 = np.tan(the_rad) * x0

    x1 = Mouth_scale * (x1 - x0) + x0
    y1 = Mouth_scale * (y1 - y0) + y0
    b = y1 - np.tan(the_rad) * x1
    x0 = b / (np.tan(the_rad + np.pi / 2) - np.tan(the_rad))
    y0 = np.tan(the_rad + np.pi / 2) * x0

    evtn = np.array([[Mouth_scale * (x1 - x0) + x0 + ix0,
                      Mouth_scale * (y1 - y0) + y0 + iy0]])
    ivtn = evtn.copy()
    ivtn[0, 1] -= Mouth_scale * v_lip[2]

    evtn = np.vstack((evtn, [evtn[0, 0] - Mouth_scale * v_lip[1], evtn[0, 1]]))
    ivtn = np.vstack((ivtn, [evtn[1, 0], ivtn[0, 1]]))

    # Sagittal assembly.
    ivt = np.vstack((ivt4, ivtn))
    evt = np.vstack((evt4, evtn))
    sagittal = np.vstack((np.flipud(ivt), evt))

    # Metrics.
    x56, y56 = sagittal[55]
    x42, y42 = sagittal[41]
    x29_30 = np.mean(sagittal[28:30], axis=0)

    gui['PD'] = np.hypot(x42 - x56, y42 - y56)
    gui['LH'] = np.hypot(x29_30[0] - x42, x29_30[1] - y42)
    gui['LHI'] = gui['LH'] / gui['PD']

    # Flattening and rotation.
    if Flat_P != 0 or Flat_T != 0:
        x50, y50 = sagittal[49]
        x9, y9 = sagittal[8]
        x44, y44 = sagittal[43]
        x15, y15 = sagittal[14]

        A1 = np.linalg.lstsq([[x44, 1], [x15, 1]], [y44, y15], rcond=None)[0]
        A2 = np.linalg.lstsq([[x50, 1], [x9, 1]], [y50, y9], rcond=None)[0]

        A = np.array([[-A1[0], 1], [-A2[0], 1]])
        B = np.array([[A1[1]], [A2[1]]])
        intersec = np.linalg.lstsq(A, B, rcond=None)[0]

        if Flat_P != 0:
            eps = np.arange(0, np.pi, np.pi / 14)
            sagpalais = sagittal[41:56]
            vect_AI = np.column_stack((intersec[0] - sagpalais[:, 0],
                                       intersec[1] - sagpalais[:, 1]))
            sagittal[41:56] += Flat_P * np.column_stack((np.sin(eps) * vect_AI[:, 0],
                                                         np.sin(eps) * vect_AI[:, 1]))

        if Flat_T != 0:
            eps = np.arange(0, np.pi, np.pi / 15)
            saglangue = sagittal[2:17]
            vect_AI = np.column_stack((intersec[0] - saglangue[:, 0],
                                       intersec[1] - saglangue[:, 1]))
            sagittal[2:17] += Flat_T * np.column_stack((np.sin(eps) * vect_AI[:, 0],
                                                        np.sin(eps) * vect_AI[:, 1]))

    if phi != 0:
        phi_rad = np.radians(phi)
        cphi = np.array([x42 + 50, y42 + 50])

        sagintrot = sagittal[:17]
        sagmiddle = sagittal[17:41]
        sagextrot = sagittal[41:]

        rot = np.array([[np.cos(phi_rad), -np.sin(phi_rad)],
                        [np.sin(phi_rad), np.cos(phi_rad)]])
        sagintrot = (sagintrot - cphi) @ rot + cphi
        sagextrot = (sagextrot - cphi) @ rot + cphi

        sagittal = np.vstack((sagintrot, sagmiddle, sagextrot))

    # Save the sagittal contour.
    gui['inci'] = sagittal[55]
    gui['sagittal'] = sagittal

    # Area-function computation.
    alpha = np.array([1.8] * 13 + [1.7] * 9 + [1.8, 1.8, 1.9, 2.0, 2.6])
    beta = np.array([1.2] * 12 + [1.3] + [1.4] * 2 + [1.5] * 12)

    c = AF_correc * vp_map
    cc = c * c

    NP = 29
    ivt = np.flipud(sagittal[:NP])
    evt = sagittal[NP:58]

    nb_permut = len(ivt) - 1
    ivt2 = np.roll(ivt, nb_permut, axis=0)
    evt2 = np.roll(evt, nb_permut, axis=0)

    p = np.hypot(ivt[:, 0] - ivt2[:, 0], ivt[:, 1] - ivt2[:, 1])[:27]
    q = np.hypot(evt[:, 0] - evt2[:, 0], evt[:, 1] - evt2[:, 1])[:27]
    r = np.hypot(evt[:, 0] - ivt[:, 0], evt[:, 1] - ivt[:, 1])[:27]
    s = np.hypot(ivt2[:, 0] - evt2[:, 0], ivt2[:, 1] - evt2[:, 1])[:27]
    t = np.hypot(ivt[:, 0] - evt2[:, 0], ivt[:, 1] - evt2[:, 1])[:27]

    a1 = 0.5 * (p + s + t)
    a2 = 0.5 * (q + r + t)
    s1 = np.sqrt(a1 * (a1 - p) * (a1 - s) * (a1 - t))
    s2 = np.sqrt(a2 * (a2 - q) * (a2 - r) * (a2 - t))

    d = 0.5 * np.hypot(ivt2[:27, 0] + evt2[:27, 0] - ivt[:27, 0] - evt[:27, 0],
                       ivt2[:27, 1] + evt2[:27, 1] - ivt[:27, 1] - evt[:27, 1])
    w = c * (s1 + s2) / d

    af = np.zeros((29, 2))
    af[:27, 0] = c * d
    af[:27, 1] = 1.4 * alpha * (w ** beta)

    lip_h = Mouth_scale * v_lip[2] / 2
    lip_w = Mouth_scale * v_lip[3] / 2
    af[27:, 0] = (ivt[NP - 2, 0] - ivt[NP - 1, 0]) * c
    af[27:, 1] = np.pi * lip_h * lip_w * cc

    af[af[:, 0] <= 0, 0] = 0.01

    gui['B'] = lip_h * c
    gui['A'] = lip_w * c
    gui['area'] = af

    return gui


# ---------------------------------------------------------------------------
# Forward-model wrappers
# ---------------------------------------------------------------------------
def forward_vlam(prm_vector, gui):
    """Update the VLAM state with a 7-parameter vector and return the new state.

    The parameter vector is placed in the first entries of ``gui['prm']``
    and the VLAM forward model is evaluated. The input state is not
    modified.
    """
    prm_vector = np.asarray(prm_vector, dtype=float).flatten()
    gui = gui.copy()
    gui['prm'][0:len(prm_vector)] = prm_vector
    return vlam2009_nn(gui)


def freqeval_formants(Ap, gui, valrect):
    """Evaluate the transfer function and first three formants for one frame.

    Updates the VLAM state with ``Ap``, rectifies the area function with
    ``softrect(., valrect)``, computes the oral transfer function on
    500 points up to 10 kHz and extracts F1, F2, F3.

    Returns ``(F1, F2, F3, gui, spectrum)``.
    """
    gui['prm'][:7] = Ap
    gui = vlam2009_nn(gui)

    area = np.column_stack((gui['area'][:, 0], softrect(gui['area'][:, 1], valrect)))

    nbfreq = 500
    Fmax = 10000
    Fmin = Fmax / (nbfreq - 1)

    H_eq, F_form = vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, 0)

    spec = np.abs(H_eq)

    F1 = F_form[0]
    F2 = F_form[1]
    F3 = F_form[2]
    return F1, F2, F3, gui, spec


# ---------------------------------------------------------------------------
# Speech synthesis from the spectral frames
# ---------------------------------------------------------------------------
def adjust_env_to_sig(env, sig):
    """Adapt envelope length to the signal length (crop or edge-pad)."""
    len_env = len(env)
    len_sig = len(sig)

    if len_env == len_sig:
        return env
    if len_env > len_sig:
        return env[:len_sig]
    return np.pad(env, (0, len_sig - len_env), mode='edge')


def synthsig(matspec, envsig, fs=20000):
    """Synthesize speech from successive spectral envelopes and an energy envelope.

    Each spectral frame (500 samples up to 10 kHz) is converted to a
    30th-order LPC filter; a glottal pulse train with a jittered F0
    trajectory (100-130-110-90 Hz cubic interpolation) excites a
    time-varying lattice filter. The result is shaped by the envelope of
    the real recording, high-emphasis filtered and normalized.
    """
    lg, _ = matspec.shape

    M = lg
    p = 30

    T_s = 2 * 5e-3
    L_t = M * T_s
    N_t = int(np.fix(L_t * fs))

    A = []
    for k in range(lg):
        _, Ai = Hfreq2lpc(matspec[k, :], p)
        A.append(Ai.flatten())

    F0 = interp1d([1, int(np.fix(N_t / 3)), int(2 * np.fix(N_t / 3)), N_t],
                  np.array([100, 130, 110, 90]),
                  kind='cubic')(np.arange(1, N_t + 1))

    np.random.seed(0)
    F0 = F0 + np.random.randn(*F0.shape) * np.mean(F0) / 100 * 0.5
    e = gen_src_3(N_t, fs, F0, 0)
    sig = f_lpc_exc2sig(e, T_s, fs, np.array(A))

    envsig = adjust_env_to_sig(envsig, sig)
    sig = lfilter([1, -0.9375], 1, envsig * sig)
    sig = sig / (1.01 * np.max(np.abs(sig)))

    return sig


def syntwav(gui, Pval, envsig, valrect):
    """Full frame-by-frame synthesis: transfer functions + formants + waveform."""
    f1, f2, f3 = [], [], []
    matspec = []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqeval_formants(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)

    f = np.array([f1, f2, f3])
    sig = synthsig(np.array(matspec), envsig)
    return sig, f


def syntform(gui, Pval, valrect):
    """Formant/transfer-function computation only (no waveform synthesis)."""
    f1, f2, f3 = [], [], []
    matspec = []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqeval_formants(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)

    f = np.array([f1, f2, f3])
    return f


# ---------------------------------------------------------------------------
# Envelope extraction from real recordings
# ---------------------------------------------------------------------------
def extract_env(audio_data, fs, target_fs=20000):
    """Extract the smoothed energy envelope of a recording.

    Hilbert envelope -> 8 Hz 4th-order Butterworth low-pass ->
    normalization to [0, 1] -> resampling to ``target_fs``.
    """
    analytic = hilbert(audio_data)
    env = np.abs(analytic)

    nyq = 0.5 * fs
    normal_cutoff = 8 / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    env_lp = filtfilt(b, a, env)

    env_lp_norm = env_lp - np.min(env_lp)
    env_lp_norm /= np.max(env_lp_norm)

    N_target = int(len(env_lp_norm) * target_fs / fs)
    env_resampled = resample(env_lp_norm, N_target)

    return env_resampled


# ---------------------------------------------------------------------------
# Parameter file readers and resampler
# ---------------------------------------------------------------------------
def read_params_npy_6(path):
    """Load a (N, 6) parameter array and append a zero Hyoid column -> (N, 7)."""
    arr = np.load(str(path), allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"File {path} must contain an array of shape (N,6). Got {arr.shape}")
    arr = arr.astype(float)
    hy_col = np.zeros((arr.shape[0], 1), dtype=float)
    return np.hstack([arr, hy_col])


def read_params_npy_7(path):
    """Load a transposed (7, N) parameter array, zero the Hyoid row -> (N, 7)."""
    arr = np.load(str(path), allow_pickle=False).T
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"File {path} must contain an array of shape (N,7). Got {arr.shape}")
    arr = arr.astype(float)
    arr[:, 6] = 0.0
    return arr


def resample_to_fs(data, fs_in, fs_out):
    """Linearly resample parameter trajectories from ``fs_in`` to ``fs_out`` Hz."""
    if fs_in == fs_out:
        return data.copy()
    N_in = data.shape[0]
    N_out = int(round(N_in * float(fs_out) / float(fs_in)))
    if N_out < 1:
        raise ValueError("N_out < 1 after resampling - check fs_in/fs_out and data length")
    t_in = np.linspace(0.0, 1.0, N_in)
    t_out = np.linspace(0.0, 1.0, N_out)
    data_out = np.empty((N_out, data.shape[1]), dtype=data.dtype)
    for c in range(data.shape[1]):
        data_out[:, c] = np.interp(t_out, t_in, data[:, c])
    return data_out


# ---------------------------------------------------------------------------
# Rendering (matplotlib frames -> OpenCV AVI)
# ---------------------------------------------------------------------------
def showgui(gui):
    """Render the midsagittal contour of a VLAM state into a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    mx = -0.0153
    my = 138.4566
    sag = gui['sagittal'] / 29.5
    h1, = ax.plot((sag[:, 0] - sag[55, 0] + mx), (sag[:, 1] + sag[55, 1]) - my, '-b')
    h1.set_color('g')
    h1.set_linewidth(2)
    ax.text(-0.5, -7.5, 'VLAM Display/GIPSA-LAB', fontsize=18)
    ax.axis([-1.5, 8.5, -7.5, 2.5])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.tight_layout()
    return fig


def render_frame(state):
    """Render one VLAM state into an RGB uint8 image array."""
    fig = showgui(state)
    fig.set_dpi(100)
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.array(buf, dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    return img


def render_video(output_video, fps, gui, Pval):
    """Render the VLAM animation for each row of ``Pval`` and write an AVI file.

    Pre-computes all articulatory states, then renders each frame with
    matplotlib (Agg backend, 100 dpi, 6x4 inch figures) and writes with
    the OpenCV XVID VideoWriter.
    """
    if Pval is None or len(Pval) == 0:
        raise ValueError("Pval is empty. Nothing to render.")

    print("Pre-computing articulatory states...")
    gui_states = []
    current_gui = gui
    for k in range(len(Pval)):
        current_gui = forward_vlam(Pval[k, :], current_gui)
        gui_states.append(current_gui.copy())

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None

    print(f"Generating {len(gui_states)} frames...")
    for idx, state in enumerate(gui_states):
        img_rgb = render_frame(state)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if video_writer is None:
            height, width = img_bgr.shape[:2]
            video_writer = cv2.VideoWriter(str(output_video), fourcc, float(fps),
                                           (width, height))
            print(f"Video initialized: {width}x{height}")

        video_writer.write(img_bgr)
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(gui_states)}")

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {output_video}")


def mux_audio_ffmpeg(video_path, audio_path, output_path, ffmpeg_exe=None):
    """Copy the video stream and add an audio track with ffmpeg."""
    import subprocess

    cmd = [
        ffmpeg_exe if ffmpeg_exe is not None else FFMPEG,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

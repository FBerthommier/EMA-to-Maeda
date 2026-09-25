"""Acoustic tube model: area function -> transfer function -> formants.

Refactored from SynthsylShort/synthSYLVcourtes.py.

Implements the lossless/lossy acoustic tube chain inherited from the ICP
(Grenoble) Fortran code (nraph3, Hugo Sanchez / Pierre Badin) as ported to
Python:
- softrect: soft rectification of section areas,
- poly2rc / levinson_durbin / Hfreq2lpc: LPC computations,
- spectrelec: electrical analogue of a series of tube sections,
- nraph_oral: Newton search for a pole of the transfer function,
- aire2spectre_oral / aire2spectre_cor_oral: area function to spectrum,
- vtn2frm_ftr_oral: vocal tract to first formants,
- freqevalNN: one articulatory vector -> F1, F2, F3 and spectrum.
"""

import numpy as np
from scipy.signal import find_peaks

from synthesis.config import (
    AIR_DENSITY, FMAX_SPEC, HEAT_RATIO, NB_FREQ, SPEED_OF_SOUND,
    SPECIFIC_HEAT_CP, WALL_DAMPING_BP, WALL_LOSS_LAMBDA, WALL_MASS_MP,
    VISCOSITY_MU,
)


def softrect(x, S):
    """Soft rectification: (sqrt(x^2 + S) + x) / 2."""
    return (np.sqrt(x ** 2 + S) + x) / 2


def poly2rc(a):
    """Convert LPC coefficients (a) to reflection coefficients (RC).

    Uses the inverse Levinson recurrence. The first element of ``a`` must be 1.
    """
    if a[0] != 1:
        raise ValueError("The first coefficient of a must be 1.")

    p = len(a) - 1          # LPC order
    k = np.zeros(p)         # Reflection coefficients
    a_current = np.copy(a[1:])  # Skip the leading 1

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
        LPC coefficients with a leading 1 (as in MATLAB).
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

        e *= (1 - k ** 2)
        if e < 0:
            e = 0

    a[1:] = -a[1:]
    return np.float64(a), e


def Hfreq2lpc(H, p):
    """Convert a frequency-response sampling to an LPC model.

    Parameters
    ----------
    H : ndarray
        Spectrum samples (N positive frequencies from 0 to (N-1)/N*fs/2).
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

    # Complete with the "negative frequencies" (conjugate symmetry).
    Hr = np.concatenate([H, [H[N - 1]], np.flipud(H[1:N - 1])])

    # Autocorrelation coefficients = IFFT of the power spectrum.
    R = np.float64(np.real(np.fft.ifft(np.abs(Hr) ** 2)))
    R = R[:p + 1]

    a, e = levinson_durbin(R, p)
    g = np.sqrt(e)

    return g, np.float64(a)


def spectrelec(w, A, zr, l, no_vibration, CONST_DAT):
    """Electrical-analogue transfer function of a chain of tube sections."""
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
    G_coef = (eta - 1) / (ro * c ** 2) * np.sqrt(lambda_ * w / (2 * cp * ro))
    R = S * l / (A ** 2) * R_coef
    G = S * l * G_coef

    YP_coef = 1 / (bp ** 2 + mp ** 2 * w ** 2)
    YP = S * l * ((bp - 1j * mp * w) * YP_coef)

    if no_vibration == 1:
        YP = 0

    Z = R + 1j * L * w
    Y = G + 1j * C * w + YP
    aa = (1 + (Z * Y / 2))

    bb = -(Z + Z ** 2 * Y / 4)
    cc = -Y
    dd = aa
    aaa = aa[0, :]
    bbb = bb[0, :]
    ccc = cc[0, :]
    ddd = dd[0, :]

    H = None
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


def nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration, CONST_DAT):
    """Newton search for one formant (pole) of the oral transfer function.

    Port of nraph3.ftn (ICP Grenoble, H. Sanchez / P. Badin).
    """
    DELTAS = complex(30, 30)
    SEUIL = 0.3
    ITER = 1
    SI = complex(-BNPE * np.pi, FE.item() * 2 * np.pi)

    F = np.nan
    BP = np.nan
    while ITER < ITERMX:
        FIcx = -1j * SI / (2 * np.pi)
        Q = 1 / aire2spectre_cor_oral(area, 1, FIcx, FIcx, F_form, NF, no_vibration, CONST_DAT)
        SIP = SI + DELTAS
        FIPcx = -1j * SIP / (2 * np.pi)
        QP = 1 / aire2spectre_cor_oral(area, 1, FIPcx, FIPcx, F_form, NF, no_vibration, CONST_DAT)
        Q1D = (QP - Q) / DELTAS
        SISU = SI - (Q / Q1D)
        if abs(SISU - SI) < SEUIL:
            F = np.imag(SISU) / (2 * np.pi)
            BP = -np.real(SISU) / np.pi
            return F, BP
        SI = SISU
        ITER += 1

    return np.float64(F), np.float64(BP)


def aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT):
    """Oral-tract area function to transfer function on a frequency grid."""
    c = CONST_DAT[0]
    rho = CONST_DAT[1]
    f = np.linspace(Fmin, Fmax, nbfreq)
    w = np.squeeze(2 * np.pi * f)

    Zr_oral = rho / (2 * np.pi * c) * (w ** 2) + \
        1j * 8 * rho / (3 * np.pi * np.sqrt(np.pi * area[-1, 1])) * w
    # no_vibration = 0 keeps wall vibration.
    H_oral = spectrelec(w, area[:, 1].reshape(-1, 1), Zr_oral,
                        area[:, 0].reshape(-1, 1), no_vibration, CONST_DAT)
    return H_oral


def aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration, CONST_DAT):
    """Transfer function corrected by the formants already found."""
    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT)

    f = np.linspace(Fmin, Fmax, nbfreq)
    if np.isrealobj(f):
        SI = 1j * 2 * np.pi * f
    else:
        SI = 2 * np.pi * 1j * f

    for I in range(NF):
        SI1 = complex(F_form[I, 2] * np.pi, F_form[I, 1] * 2 * np.pi)
        H_eq *= np.squeeze(((SI - SI1) * (SI - np.conj(SI1))) / (SI1 * np.conj(SI1)))

    return H_eq


def vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT):
    """Vocal tract area function to the first formants (frequency, bandwidth).

    Iteratively extracts all poles below Fmax with the Newton search, then
    returns the spectrum and the first formants.
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
        H_eq = aire2spectre_cor_oral(area, nbfreq, Fmax, Fmin, F_form, NF, no_vibration, CONST_DAT)
        ind_maxi, _ = find_peaks(np.asarray(20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        ind_mini, _ = find_peaks(np.asarray(-20 * np.log10(np.abs(H_eq)), dtype=np.float64).flatten())
        frq_min = f[ind_mini]
        frq_max = f[ind_maxi]
        FEST = frq_max
        if any(FEST):
            FE = FEST[0]

        F, BP = nraph_oral(FE, BNPE, ITERMX, FMAX, area, F_form, NF, no_vibration, CONST_DAT)
        F_form[NF, :] = [0, F.item(), BP.item()]
        NF += 1

    H_eq = aire2spectre_oral(area, nbfreq, Fmax, Fmin, no_vibration, CONST_DAT)

    nbformants = 3
    F_form1 = np.zeros((nbformants, 1))
    for k in range(min(nbformants, NF)):
        F_form1[k] = F_form[k, 1]

    return H_eq, F_form1


def get_const_dat():
    """Assemble the physical-constant vector used by the tube model."""
    return np.array([SPEED_OF_SOUND, AIR_DENSITY, WALL_LOSS_LAMBDA, HEAT_RATIO,
                     VISCOSITY_MU, SPECIFIC_HEAT_CP, WALL_DAMPING_BP,
                     WALL_MASS_MP])


def freqevalNN(Ap, gui, valrect):
    """Articulatory vector -> (F1, F2, F3, updated gui, |H| spectrum).

    Updates the VLAM parameters, computes the area function with soft
    rectification ``valrect`` and runs the tube model to get the formants.
    """
    CONST_DAT = get_const_dat()

    gui["prm"][:7] = Ap
    from synthesis.vlam import vlam2009NN  # local import to avoid a cycle
    gui = vlam2009NN(gui)

    area = np.column_stack((gui["area"][:, 0], softrect(gui["area"][:, 1], valrect)))

    Fmax = FMAX_SPEC
    nbfreq = NB_FREQ
    Fmin = Fmax / (nbfreq - 1)

    H_eq, F_form = vtn2frm_ftr_oral(area, nbfreq, Fmax, Fmin, 0, CONST_DAT)

    spec = np.abs(H_eq)

    F1 = F_form[0]
    F2 = F_form[1]
    F3 = F_form[2]
    return F1, F2, F3, gui, spec


def synthpause(gui, Pval, valrect):
    """Formants F1-F3 for a pause trajectory (no audio synthesis)."""
    f1, f2, f3 = [], [], []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, _, _ = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)

    f = np.array([f1, f2, f3]).T
    return f

"""Articulatory trajectory generation with the Tau model.

Refactored from SynthsylShort/synthSYLVcourtes.py.

Movement targets live in polar coordinates [rho, theta] on a movement circle
described by ``CO``; ``arc`` interpolates between two targets along circular
arcs whose speed profile is governed by the Tau model (cosine power profile
with exponent Pexp). ``boucle``/``boucleplot`` assemble the full parameter
trajectory for each syllable type, and ``parse`` classifies characters of a
syllable into vowels and consonants.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Consonant target positions
# ---------------------------------------------------------------------------
def consvalD(thetavoy):
    """Theta target of the denti-alveolar consonant [d]."""
    return -4.5 * np.pi / 8


def consvalG(thetavoy):
    """[rho, theta] target of the velar consonant [g], context-dependent."""
    if thetavoy <= np.pi:    # < /ga/ palatal
        return [1.2, np.pi / 3]
    return [1.1, -np.pi / 12]


def consvalD2(thetavoy):
    """Theta target of the denti-alveolar consonant in cluster context."""
    return -2.5 * np.pi / 6


def consvalG2():
    """Theta target of the velar consonant in cluster context."""
    return -np.pi / 12


# ---------------------------------------------------------------------------
# Arc interpolation
# ---------------------------------------------------------------------------
def _arc_thetas(thetabounds, D, opint):
    """Theta samples of an arc: opint 0 = closed-closed, -1 = open-closed,
    1 = closed-open."""
    if opint == 0:      # closed-closed
        return np.linspace(thetabounds[0], thetabounds[1], D)
    if opint == -1:     # open-closed
        theta1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        return theta1[1:D + 1]
    if opint == 1:      # closed-open
        return np.linspace(thetabounds[0], thetabounds[1], D + 1)
    raise ValueError(f"Invalid opint value: {opint}")


def arc(pt, co, params, D, thetabounds, opint, nu, K, Pexp):
    """Parameter trajectories for selected articulators along a Tau arc.

    Parameters
    ----------
    pt : ndarray
        Two movement targets [rho, theta] (rows: departure, arrival).
    co : ndarray
        Movement-circle geometry [rho, offset, angle] per articulator.
    params : sequence of int
        Indices of the articulators to move.
    D : int
        Number of time frames.
    thetabounds : sequence
        Bounds of the arc on the movement circle.
    opint : int
        Endpoint inclusion mode (see ``_arc_thetas``).
    nu, K, Pexp : Tau-model parameters.
    """
    theta = _arc_thetas(thetabounds, D, opint)

    pd = pt[0, :]
    pa = pt[1, :]

    Pt = np.zeros((D, len(params)))
    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        Pt[k, :] = co[params, 1] + \
            rho * pa[0] * co[params, 0] * np.cos(co[params, 2] - pa[1] - (nu / K) * theta[k]) + \
            (1 - rho) * pd[0] * co[params, 0] * np.cos(co[params, 2] - pd[1])

    return Pt


def arcplot(pt, D, thetabounds, opint, nu, K, Pexp):
    """Complex trajectory of a two-target movement (for polar plotting)."""
    theta = _arc_thetas(thetabounds, D, opint)

    pd = pt[0, :]
    pa = pt[1, :]
    Tt = np.zeros(D, dtype=complex)

    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        Tt[k] = (rho * pa[0] * np.cos(pa[1] + (nu / K) * theta[k]) +
                 (1 - rho) * pd[0] * np.cos(pd[1])) + \
                1j * (rho * pa[0] * np.sin(pa[1] + (nu / K) * theta[k]) +
                      (1 - rho) * pd[0] * np.sin(pd[1]))

    return Tt


def arcplotV(pt, D, thetabounds, open_, nu, K, Pexp):
    """Two-column complex trajectory (both columns identical), for plotting."""
    theta = _arc_thetas(thetabounds, D, open_)

    pd = pt[0, :]
    pa = pt[1, :]
    Tt = np.zeros((D, 2), dtype=complex)

    for k in range(D):
        rho = np.cos(theta[k] / 2) ** Pexp
        Tt[k, 0] = (rho * pa[0] * np.cos(pa[1] + (nu / K) * theta[k]) +
                    (1 - rho) * pd[0] * np.cos(pd[1])) + \
                   1j * (rho * pa[0] * np.sin(pa[1] + (nu / K) * theta[k]) +
                         (1 - rho) * pd[0] * np.sin(pd[1]))

    Tt[:, 1] = Tt[:, 0]
    return Tt


# ---------------------------------------------------------------------------
# Syllable loops
# ---------------------------------------------------------------------------
def makeloop(n, sylP1, sylpt, nbP, co, T, nu, K, Kvoy, Pexp):
    """Parameter trajectory of a consonant-vowel loop.

    ``n`` = 2 or 3 movement segments; ``sylP1`` = 1-based indices of the
    moving articulators; ``sylpt`` = target points; remaining articulators
    follow a vowel-speed arc.
    """
    sylP = sylP1 - 1
    nmP = np.setxor1d(sylP, np.arange(nbP))
    Pval = np.zeros((n * T, nbP))
    if n == 2:
        Pval[:, sylP] = np.vstack(
            (arc(sylpt[[1, 0], :], co, sylP, T, np.array([0, np.pi]), 0, nu, K, Pexp),
             arc(sylpt[[1, 2], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp)))
        if nmP.size > 0:
            Pval[:, nmP] = arc(sylpt[[0, 2], :], co, nmP, n * T, np.array([-np.pi, 0]), 0, nu, Kvoy, Pexp)

    if n == 3:
        Pval[:, sylP] = np.vstack(
            (arc(sylpt[[1, 0], :], co, sylP, T, np.array([0, np.pi]), 0, nu, K, Pexp),
             arc(sylpt[[1, 2], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp),
             arc(sylpt[[2, 3], :], co, sylP, T, np.array([-np.pi, 0]), -1, nu, K, Pexp)))
        if nmP.size > 0:
            Pval[:, nmP] = arc(sylpt[[0, 3], :], co, nmP, n * T, np.array([-np.pi, 0]), 0, nu, Kvoy, Pexp)
    return Pval


def makelooplot(n, sylpt, T, nu, K, Kvoy, Pexp):
    """Complex polar trajectory of a CV loop, for plotting."""
    Tval = np.zeros((n * T, 2), dtype=complex)
    if n == 2:
        Tval[:, 0] = np.concat(
            (arcplot(sylpt[[1, 0], :], T, [0, np.pi], 0, nu, K, Pexp),
             arcplot(sylpt[[1, 2], :], T, [-np.pi, 0], -1, nu, K, Pexp)))
        Tval[:, 1] = arcplot(sylpt[[0, 2], :], n * T, [-np.pi, 0], 0, nu, Kvoy, Pexp)
    if n == 3:
        Tval[:, 0] = np.concat(
            (arcplot(sylpt[[1, 0], :], T, [0, np.pi], 0, nu, K, Pexp),
             arcplot(sylpt[[1, 2], :], T, [-np.pi, 0], -1, nu, K, Pexp),
             arcplot(sylpt[[2, 3], :], T, [-np.pi, 0], -1, nu, K, Pexp)))
        Tval[:, 1] = arcplot(sylpt[[0, 3], :], n * T, [-np.pi, 0], 0, nu, Kvoy, Pexp)
    return Tval


def boucle(syl, co, nu, K, Kvoy, T, Pexp):
    """Articulatory parameter trajectory for one syllable dict."""
    nbP, _ = co.shape
    typ = syl["typ"].upper()
    Pval = None

    if typ == "CV":
        Pval = np.vstack((makeloop(2, syl["P"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),))
    elif typ == "CVV":
        Pval = np.vstack((makeloop(2, syl["P"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCVV":
        Pval = np.vstack((makeloop(2, syl["P"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VC":
        Pval = np.vstack((makeloop(2, syl["P"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),))
    elif typ == "VVC":
        Pval = np.vstack((arc(syl["pt"][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makeloop(2, syl["P"], syl["pt"][1:4, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "VCV":
        Pval = np.vstack((makeloop(2, syl["P"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),))
    elif typ == "CVC":
        Pval = np.vstack((makeloop(2, syl["P1"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          makeloop(2, syl["P2"], syl["pt"][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "VCVC":
        Pval = np.vstack((makeloop(2, syl["P1"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          makeloop(2, syl["P2"], syl["pt"][2:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "CVVC":
        Pval = np.vstack((makeloop(2, syl["P1"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makeloop(2, syl["P2"], syl["pt"][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "VCVVC":
        Pval = np.vstack((makeloop(2, syl["P1"], syl["pt"][0:3, :], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][2:4, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makeloop(2, syl["P2"], syl["pt"][3:6, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "CCV":
        Pval = np.vstack((makeloop(3, syl["P"], syl["pt"], nbP, co, T, nu, K, Kvoy, Pexp),))
    elif typ == "CCVV":
        Pval = np.vstack((makeloop(3, syl["P"], syl["pt"], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCCVV":
        Pval = np.vstack((makeloop(3, syl["P"], syl["pt"], nbP, co, T, nu, K, Kvoy, Pexp),
                          arc(syl["pt"][3:5, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCC":
        Pval = np.vstack(makeloop(3, syl["P"], syl["pt"], nbP, co, T, nu, K, Kvoy, Pexp))
    elif typ == "VVCC":
        Pval = np.vstack((arc(syl["pt"][0:2, :], co, np.arange(0, nbP), 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makeloop(3, syl["P"], syl["pt"][1:5, :], nbP, co, T, nu, K, Kvoy, Pexp)))
    elif typ == "VCCV":
        Pval = np.vstack(makeloop(3, syl["P"], syl["pt"], nbP, co, T, nu, K, Kvoy, Pexp))
    else:
        raise ValueError(f"Unknown syllable type: {typ}")

    return np.array(Pval)


def boucleplot(syl, nu, K, Kvoy, T, Pexp):
    """Complex polar plotting trajectory for one syllable dict."""
    typ = syl["typ"].upper()

    if typ == "CV":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),))
    elif typ == "CVV":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCVV":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VC":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),))
    elif typ == "VVC":
        Tval = np.vstack((arcplotV(syl["pt"][0:2, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makelooplot(2, syl["pt"][1:4, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "VCV":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),))
    elif typ == "CVC":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          makelooplot(2, syl["pt"][2:5, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "VCVC":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          makelooplot(2, syl["pt"][2:5, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "CVVC":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makelooplot(2, syl["pt"][3:6, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "VCVVC":
        Tval = np.vstack((makelooplot(2, syl["pt"][0:3, :], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][2:4, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makelooplot(2, syl["pt"][3:6, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "CCV":
        Tval = np.vstack((makelooplot(3, syl["pt"], T, nu, K, Kvoy, Pexp),))
    elif typ == "CCVV":
        Tval = np.vstack((makelooplot(3, syl["pt"], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][3:5, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCCVV":
        Tval = np.vstack((makelooplot(3, syl["pt"], T, nu, K, Kvoy, Pexp),
                          arcplotV(syl["pt"][3:5, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp)))
    elif typ == "VCC":
        Tval = np.vstack((makelooplot(3, syl["pt"], T, nu, K, Kvoy, Pexp),))
    elif typ == "VVCC":
        Tval = np.vstack((arcplotV(syl["pt"][0:2, :], 2 * T, [-np.pi, 0], 0, nu, K, Pexp),
                          makelooplot(3, syl["pt"][1:5, :], T, nu, K, Kvoy, Pexp)))
    elif typ == "VCCV":
        Tval = np.vstack((makelooplot(3, syl["pt"], T, nu, K, Kvoy, Pexp),))
    else:
        raise ValueError(f"Unknown syllable type: {typ}")
    return Tval


# ---------------------------------------------------------------------------
# Syllable parsing
# ---------------------------------------------------------------------------
def parse(C, tabvoy, tabcons, rho, theta):
    """Classify the characters of each syllable string in ``C``.

    Parameters
    ----------
    C : list of str
        Syllable strings.
    tabvoy : list of str
        Vowel characters.
    tabcons : list of str
        Consonant characters.
    rho, theta : lists
        Polar target of each vowel.

    Returns
    -------
    booldeb : list
        Syllable onset indicator (1 = consonant, 0 = vowel, -1 = no vowel).
    tabdeb : list
        Theta of the first vowel of each syllable (or -1).
    boolast : list
        Syllable offset indicator (1 = consonant, 0 = vowel, -1 = no vowel).
    tablast : list
        [theta, rho] of the last vowel of each syllable (or [-1, -1]).
    """
    booldeb = []
    tabdeb = []
    boolast = []
    tablast = []

    for R in C:
        typ = list(R)

        # Locate consonants.
        for cons in tabcons:
            for i, ch in enumerate(R):
                if ch == cons:
                    typ[i] = cons.upper()
        # Locate vowels.
        for voy in tabvoy:
            for i, ch in enumerate(R):
                if ch == voy:
                    typ[i] = "V"

        locvoy = [i for i, ch in enumerate(typ) if ch == "V"]

        if locvoy:
            first, last = locvoy[0], locvoy[-1]
            tabdeb.append(theta[tabvoy.index(R[first])])
            booldeb.append(0 if typ[0] == "V" else 1)
            tablast.append([theta[tabvoy.index(R[last])],
                            rho[tabvoy.index(R[last])]])
            boolast.append(0 if typ[-1] == "V" else 1)
        else:
            booldeb.append(-1)
            tabdeb.append(-1)
            boolast.append(-1)
            tablast.append([-1, -1])

    return booldeb, tabdeb, boolast, tablast

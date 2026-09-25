"""Syllable dictionary construction from character strings.

Refactored from SynthsylShort/synthSYLVcourtes.py (the large type-dispatch
block of main()).

Turns a syllable string such as 'ba', 'aby' or 'badu' into a dict with:
- 'typ': canonical CV skeleton ('CV', 'VVC', 'CcV', ...),
- 'pt': polar movement targets [rho, theta],
- 'P'/'P1'/'P2': 1-based articulator indices moved by the consonants.
"""

import numpy as np

from synthesis.config import (
    ART, ART1, COEFCEN, RHO, THETA, VOWELS, CONSONANTS,
)
from synthesis.trajectories import consvalD, consvalD2, consvalG, consvalG2


def syllable_types(R, tabvoy=VOWELS, tabcons=CONSONANTS):
    """Return the canonical CV skeleton of a syllable string.

    Consonant characters are upper-cased, vowel characters become 'V'.
    """
    typ = list(R)

    for conso in tabcons:
        loccons = [i for i in range(len(R)) if R[i] == conso]
        for loc in loccons:
            typ[loc] = conso.upper()

    for voy in tabvoy:
        locvoy = [i for i in range(len(R)) if R[i] == voy]
        for loc in locvoy:
            typ[loc] = "V"

    return "".join(typ)


def build_syllable(R, tabvoy=VOWELS, tabcons=CONSONANTS,
                   rho=None, theta=None, voydeb=None):
    """Build the syllable dict for string ``R``.

    Parameters
    ----------
    R : str
        Syllable characters (e.g. 'ba', 'aby', 'stra' are not supported).
    tabvoy, tabcons : lists
        Vowel and consonant inventories.
    rho, theta : lists
        Polar targets of the vowels.
    voydeb : [rho, theta]
        Departure vowel target (coming from the previous syllable or default).

    Returns
    -------
    syl : dict
        Syllable description (see module docstring).
    ok : bool
        False if the syllable structure is unknown.
    """
    if rho is None:
        rho = RHO
    if theta is None:
        theta = THETA

    typ = syllable_types(R, tabvoy, tabcons)

    n1 = [-1] * len(typ)
    for k, conso in enumerate(tabcons):
        for loc in range(len(R)):
            if R[loc] == conso:
                n1[loc] = k

    k1 = [-1] * len(typ)
    for k, voy in enumerate(tabvoy):
        for loc in range(len(R)):
            if R[loc] == voy:
                k1[loc] = k

    # Default consonant targets; entries 1 and 2 are overwritten per context.
    cons = np.array([[1.2, np.pi / 3],
                     [1.2, 0],
                     [1.1, 0]])

    ok = True
    if typ in ["BV", "DV", "GV"]:
        cons[1, 1] = consvalD(theta[int(k1[1])])
        cons[2, :] = consvalG(theta[int(k1[1])])
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]]]),
            "P": ART[n1[0]][ART[n1[0]] != 0],
            "typ": "CV"}
    elif typ in ["BVV", "DVV", "GVV"]:
        cons[1, 1] = consvalD(theta[int(k1[1])])
        cons[2, :] = consvalG(theta[int(k1[1])])
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]],
                             [rho[k1[2]], theta[k1[2]]]]),
            "P": ART[n1[0]][ART[n1[0]] != 0],
            "typ": "CVV"}
    elif typ in ["VBVV", "VDVV", "VGVV"]:
        cons[1, 1] = consvalD(theta[int(k1[2])])
        cons[2, :] = consvalG(theta[int(k1[2])])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]],
                             [rho[k1[3]], theta[k1[3]]]]),
            "P": ART[n1[1]][ART[n1[1]] != 0],
            "typ": "VCVV"}
    elif typ in ["VB", "VD", "VG"]:
        cons[1, 1] = consvalD(theta[int(k1[0])])
        cons[2, :] = consvalG(theta[int(k1[0])])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]],
                             [COEFCEN * rho[k1[0]], theta[k1[0]]]]),
            "P": ART[n1[1]][ART[n1[1]] != 0],
            "typ": "VC"}
    elif typ in ["VVB", "VVD", "VVG"]:
        cons[1, 1] = consvalD(theta[int(k1[0])])
        cons[2, :] = consvalG(theta[int(k1[0])])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]],
                             [rho[k1[1]], theta[k1[1]]], cons[n1[2]],
                             [COEFCEN * rho[k1[1]], theta[k1[1]]]]),
            "P": ART[n1[2]][ART[n1[2]] != 0],
            "typ": "VVC"}
    elif typ in ["VBV", "VDV", "VGV"]:
        cons[1, 1] = consvalD(theta[int(k1[2])])
        cons[2, :] = consvalG(theta[int(k1[2])])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]]]),
            "P": ART[n1[1]][ART[n1[1]] != 0],
            "typ": "VCV"}
    elif typ in ["BVB", "BVD", "BVG", "DVB", "DVD", "DVG", "GVB", "GVD", "GVG"]:
        cons[1][1] = consvalD(theta[k1[1]])
        cons[2, :] = consvalG(theta[k1[1]])
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]],
                             cons[n1[2]], [COEFCEN * rho[k1[1]], theta[k1[1]]]]),
            "P1": ART[n1[0]][ART[n1[0]] != 0],
            "P2": ART[n1[2]][ART[n1[2]] != 0],
            "typ": "CVC"
        }
    elif typ in ["BVVB", "BVVD", "BVVG", "DVVB", "DVVD", "DVVG", "GVVB", "GVVD", "GVVG"]:
        cons[1][1] = consvalD(theta[k1[1]])
        cons[2, :] = consvalG(theta[k1[1]])
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], [rho[k1[1]], theta[k1[1]]],
                             [rho[k1[2]], theta[k1[2]]], cons[n1[3]],
                             [COEFCEN * rho[k1[2]], theta[k1[2]]]]),
            "P1": ART[n1[0]][ART[n1[0]] != 0],
            "P2": ART[n1[3]][ART[n1[3]] != 0],
            "typ": "CVVC"
        }
    elif typ in ["VBVB", "VBVD", "VBVG", "VDVB", "VDVD", "VDVG", "VGVB", "VGVD", "VGVG"]:
        cons[1][1] = consvalD(theta[k1[2]])
        cons[2, :] = consvalG(theta[k1[2]])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]], cons[n1[3]],
                             [COEFCEN * rho[k1[2]], theta[k1[2]]]]),
            "P1": ART[n1[1]][ART[n1[1]] != 0],
            "P2": ART[n1[3]][ART[n1[3]] != 0],
            "typ": "VCVC"
        }
    elif typ in ["VBVVB", "VBVVD", "VBVVG", "VDVVB", "VDVVD", "VDVVG", "VGVVB", "VGVVD", "VGVVG"]:
        cons[1][1] = consvalD(theta[k1[2]])
        cons[2, :] = consvalG(theta[k1[2]])
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]], [rho[k1[3]], theta[k1[3]]],
                             cons[n1[4]], [COEFCEN * rho[k1[3]], theta[k1[3]]]]),
            "P1": ART[n1[1]][ART[n1[1]] != 0],
            "P2": ART[n1[4]][ART[n1[4]] != 0],
            "typ": "VCVVC"
        }
    elif typ in ["VBD", "VDB", "VBG", "VGB", "VDG", "VGD"]:
        cons[1][1] = consvalD2(theta[k1[0]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]],
                             [COEFCEN * rho[k1[0]], theta[k1[0]]]]),
            "P": ART1[int(n1[1] + n1[2] > 2)],
            "typ": "VcC"
        }
    elif typ in ["VVBD", "VVDB", "VVBG", "VVGB", "VVDG", "VVGD"]:
        cons[1][1] = consvalD2(theta[k1[1]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], [rho[k1[1]], theta[k1[1]]],
                             cons[n1[2]], cons[n1[3]],
                             [COEFCEN * rho[k1[1]], theta[k1[1]]]]),
            "P": ART1[int(n1[2] + n1[3] > 2)],
            "typ": "VVcC"
        }
    elif typ in ["BDV", "DBV", "BGV", "GBV", "DGV", "GDV"]:
        cons[1][1] = consvalD2(theta[k1[2]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]]]),
            "P": ART1[int(n1[0] + n1[1] > 2)],
            "typ": "CcV"
        }
    elif typ in ["VBDV", "VDBV", "VBGV", "VGBV", "VDGV", "VGDV"]:
        cons[1][1] = consvalD2(theta[k1[3]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]],
                             [rho[k1[3]], theta[k1[3]]]]),
            "P": ART1[int(n1[1] + n1[2] > 2)],
            "typ": "VCcV"
        }
    elif typ in ["BDVV", "DBVV", "BGVV", "GBVV", "DGVV", "GDVV"]:
        cons[1][1] = consvalD2(theta[k1[2]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([voydeb, cons[n1[0]], cons[n1[1]],
                             [rho[k1[2]], theta[k1[2]]], [rho[k1[3]], theta[k1[3]]]]),
            "P": ART1[int(n1[0] + n1[1] > 2)],
            "typ": "CcVV"
        }
    elif typ in ["VBDVV", "VDBVV", "VBGVV", "VGBVV", "VDGVV", "VGDVV"]:
        cons[1][1] = consvalD2(theta[k1[3]])
        cons[2][1] = consvalG2()
        syl = {
            "pt": np.vstack([[rho[k1[0]], theta[k1[0]]], cons[n1[1]], cons[n1[2]],
                             [rho[k1[3]], theta[k1[3]]], [rho[k1[4]], theta[k1[4]]]]),
            "P": ART1[int(n1[1] + n1[2] > 2)],
            "typ": "VCcVV"
        }
    else:
        print("Unknown syllable")
        syl = None
        ok = False

    return syl, ok

"""VLAM articulatory model (Maeda-style sagittal midsagittal contours).

Refactored from SynthsylShort/synthSYLVcourtes.py.

The VLAM model was written by Shinji Maeda, transcribed from C to Matlab by
David Pochic and Nassim Zga (modifications JLS/LJB: palate & pharynx; uvular
support; duct rotation; summer 2006). This module converts the 7 articulatory
parameters (jaw, tongue body, dorsum, apex, lip height, lip protrusion,
larynx) plus age/scaling parameters into sagittal contours and a 29-section
area function.
"""

import numpy as np


def initVLAMLength(Ap, L):
    """Initialize the GUI/state dictionary for a vocal tract of length ``L``.

    ``Ap`` holds the 7 articulatory parameters; ``L`` sets the age-dependent
    growth factor kage.
    """
    gui = {}
    gui["sagittal"] = np.zeros((58, 2))
    gui["inci"] = np.zeros((1, 2))
    gui["area"] = np.zeros((29, 2))
    gui["PD"] = 0
    gui["LH"] = 0
    gui["LHI"] = 0
    gui["B"] = 0
    gui["A"] = 0

    kage = 5.4749 * L - 407.1374
    gui["prm"] = np.zeros(11)
    gui["prm"][0:7] = Ap
    gui["prm"][7] = kage
    gui["prm"][8] = 0
    gui["prm"][10] = 0
    gui["prm"][9] = 0
    return gui


# Constant model matrices (tongue, lips, larynx shape functions).
A_TNG = np.array([[1.000000, 0.000000, 0.000000, 0.000000],
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

S_TNG = np.array([27.674635, 29.947931, 44.694466, 99.310226, 96.871323,
                  84.140404, 78.357513, 73.387718, 72.926758, 71.453232,
                  69.288765, 66.615509, 63.603722, 59.964859, 56.695446,
                  56.415058, 62.016468, 73.235176, 84.008438, 91.488312,
                  94.124176, 95.246323, 93.516365, 93.000343, 100.934669,
                  106.512482])

U_TNG = np.array([104.271675, 443.988434, 450.481689, 399.942200, 348.603088,
                  351.181122, 365.404633, 370.290955, 356.202301, 341.890167,
                  332.117523, 326.826599, 326.512512, 331.631989, 343.175323,
                  361.265900, 385.231201, 411.826599, 435.691711, 455.040466,
                  462.736023, 453.025055, 432.250488, 407.358368, 384.551056,
                  363.836212])

A_LIP = np.array([[1.000000, 0.000000, 0.000000],
                  [0.178244, -0.395733, 0.888897],
                  [-0.154638, 0.987971, 0.000000],
                  [-0.217332, 0.825187, -0.303429]])

S_LIP = np.array([27.674635, 33.068081, 99.392258, 213.996170])
U_LIP = np.array([104.271675, 122.812141, 135.938339, 460.440857])

A_LRX = np.array([[1.000000, 0.000000],
                  [-0.208338, 0.262446],
                  [0.127814, 0.991798],
                  [-0.131840, 0.300784],
                  [0.097688, 0.934267]])

S_LRX = np.array([27.674635, 41.593315, 65.562340, 44.372742, 66.147499])
U_LRX = np.array([104.271675, 143.138733, -948.229309, 404.678223, -962.936401])

U_WAL = np.array([550.196533, 604.878601, 674.127197, 678.776489, 665.905579,
                  653.312134, 643.223511, 633.836243, 636.994202, 668.834290,
                  703.098267, 600, 610, 610, 605, 600, 600, 600, 600, 600,
                  600, 600, 600, 600, 600])

# Area-function profile exponents.
ALPHA = np.array([1.8] * 13 + [1.7] * 9 + [1.8, 1.8, 1.9, 2.0, 2.6])
BETA = np.array([1.2] * 12 + [1.3] + [1.4] * 2 + [1.5] * 12)


def _scale_point(x1, y1, angle_rad, scale, ix0, iy0):
    """Scale a point away from the intersection of two orthogonal axes."""
    b = y1 - np.tan(angle_rad + np.pi / 2) * x1
    x0 = b / (np.tan(angle_rad) - np.tan(angle_rad + np.pi / 2))
    y0 = np.tan(angle_rad) * x0

    x1 = scale * (x1 - x0) + x0
    y1 = scale * (y1 - y0) + y0
    b = y1 - np.tan(angle_rad) * x1
    x0 = b / (np.tan(angle_rad + np.pi / 2) - np.tan(angle_rad))
    y0 = np.tan(angle_rad + np.pi / 2) * x0

    return scale * (x1 - x0) + x0 + ix0, scale * (y1 - y0) + y0 + iy0


def vlam2009NN(gui):
    """Compute sagittal contours and area function from gui['prm'].

    Input: gui['prm'][0:7] = jaw, tongue body, dorsum, apex, lip height,
    lip protrusion, larynx; gui['prm'][7] = age factor; gui['prm'][8:11] =
    pharynx flattening, duct rotation, tongue flattening.

    Updates gui['sagittal'], gui['area'], gui['PD'], gui['LH'], gui['LHI'],
    gui['inci'], gui['B'], gui['A'].
    """
    # Extract and reorder the parameters (identical to the original model).
    parametres = np.zeros(10)
    parametres[0] = gui["prm"][0]  # Jaw
    parametres[1] = gui["prm"][1]  # Body
    parametres[2] = gui["prm"][2]  # Drsm
    parametres[3] = gui["prm"][3]  # Apex
    parametres[4] = gui["prm"][5]  # LipP
    parametres[5] = gui["prm"][4]  # LipH
    parametres[6] = gui["prm"][6]  # Larynx
    parametres[7] = gui["prm"][7]  # Age

    # "Human" mode scaling coefficients.
    parametres[8] = 0.8    # k_Pharynx
    parametres[9] = 0.35   # k_Mouth

    # Flattening and rotation parameters.
    Flat_P = -gui["prm"][8]
    Flat_T = -gui["prm"][10]
    phi = gui["prm"][9]
    AF_correc = 1

    # Initial computations.
    k_age = parametres[7]
    k_age_max = 650
    Pharynx_scale = k_age * parametres[8] / k_age_max + 0.3
    Mouth_scale = k_age * parametres[9] / k_age_max + 0.65

    # Unit conversion.
    vp_map = 1 / 29.5
    TEKvt = 188.679245
    TEKvt *= vp_map

    s_tng = S_TNG / TEKvt
    u_tng = U_TNG / TEKvt
    s_lip = S_LIP / TEKvt
    u_lip = U_LIP / TEKvt
    s_lrx = S_LRX / TEKvt
    u_lrx = U_LRX / TEKvt
    u_wal = U_WAL / TEKvt

    # Contour computations.
    v_tng = s_tng * (A_TNG @ parametres[:4]) + u_tng
    v_lip = s_lip * (A_LIP @ np.array([parametres[0], parametres[4], parametres[5]])) + u_lip
    v_lip[v_lip < 0] = 0
    v_lrx = s_lrx * (A_LRX @ np.array([parametres[0], parametres[6]])) + u_lrx

    # Reference points.
    ix0, iy0 = 2200, 2000
    r, dl, m1, m2, m3 = 5, 0.5, 14, 11, 6
    omega, theta = -11.25, 11.25

    r_vp = r / vp_map
    dlPharynx_vp = Pharynx_scale * dl / vp_map
    dlPalatal_vp = Mouth_scale * dl / vp_map
    ome = np.radians(omega)
    the = np.radians(theta)

    # Grid construction: pharyngeal, palatal and labial parts.
    dx_i = dlPharynx_vp * np.cos(ome - np.pi / 2)
    dy_i = dlPharynx_vp * np.sin(ome - np.pi / 2)
    dx_e = r_vp * np.cos(ome)
    dy_e = r_vp * np.sin(ome)

    igd1 = np.column_stack((dx_i * np.arange(m1 - 1, -1, -1) + ix0,
                            dy_i * np.arange(m1 - 1, -1, -1) + iy0))
    egd1 = np.column_stack((dx_e + igd1[:, 0], dy_e + igd1[:, 1]))

    gam = the * np.arange(1, m2 + 1) + ome
    igd2 = np.column_stack((np.full(m2, ix0), np.full(m2, iy0)))
    egd2 = np.column_stack((r_vp * np.cos(gam) + ix0, r_vp * np.sin(gam) + iy0))

    dx_i = dlPalatal_vp * np.cos(gam[-1] + np.pi / 2)
    dy_i = dlPalatal_vp * np.sin(gam[-1] + np.pi / 2)
    dx_e = r_vp * np.cos(gam[-1])
    dy_e = r_vp * np.sin(gam[-1])

    igd3 = np.column_stack((dx_i * np.arange(1, m3 + 1) + ix0,
                            dy_i * np.arange(1, m3 + 1) + iy0))
    egd3 = np.column_stack((dx_e + igd3[:, 0], dy_e + igd3[:, 1]))

    igd = np.vstack((igd1, igd2, igd3))
    egd = np.vstack((egd1, egd2, egd3))

    # Normalized segment directions.
    p, q = egd[:, 0] - igd[:, 0], egd[:, 1] - igd[:, 1]
    s = np.hypot(p, q)
    vtos = np.column_stack((p / s, q / s))

    # Jaw reference points.
    omg = np.radians(omega)
    JAW = 0

    x1, y1 = v_lrx[JAW + 1], v_lrx[JAW + 2]
    ix1, iy1 = _scale_point(x1, y1, omg, Pharynx_scale, ix0, iy0)
    ivt1 = np.array([[ix1, iy1]])

    x1, y1 = v_lrx[JAW + 3], v_lrx[JAW + 4]
    ix1, iy1 = _scale_point(x1, y1, omg, Pharynx_scale, ix0, iy0)
    evt1 = np.array([[ix1, iy1]])

    # Scale factors along the tract.
    scale_factor1 = Pharynx_scale * np.ones((8, 1))
    scale_factor2 = (Mouth_scale - Pharynx_scale) * np.arange(1, m2 + 1).reshape(-1, 1) / m2 + Pharynx_scale
    scale_factor3 = Mouth_scale * np.ones((6, 1))
    scale_factor = np.vstack((scale_factor1, scale_factor2, scale_factor3)).flatten()

    # Tongue contour blocked at the walls.
    v = scale_factor[:25] * np.minimum(v_tng[1:26], u_wal)
    xy1 = igd[6:31] + vtos[6:31] * v.reshape(-1, 1)
    xy2 = igd[6:31] + vtos[6:31] * (u_wal * scale_factor[:25]).reshape(-1, 1)

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
    ix1, iy1 = _scale_point(inci_x, inci_y + inci_lip_vp, the_rad, Mouth_scale, ix0, iy0)
    evtn = np.array([[ix1, iy1]])
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

    gui["PD"] = np.hypot(x42 - x56, y42 - y56)
    gui["LH"] = np.hypot(x29_30[0] - x42, x29_30[1] - y42)
    gui["LHI"] = gui["LH"] / gui["PD"]

    # Flattening and rotation post-adjustments.
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
            vect_AI = np.column_stack((intersec[0] - sagpalais[:, 0], intersec[1] - sagpalais[:, 1]))
            sagittal[41:56] += Flat_P * np.column_stack((np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1]))

        if Flat_T != 0:
            eps = np.arange(0, np.pi, np.pi / 15)
            saglangue = sagittal[2:17]
            vect_AI = np.column_stack((intersec[0] - saglangue[:, 0], intersec[1] - saglangue[:, 1]))
            sagittal[2:17] += Flat_T * np.column_stack((np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1]))

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

    gui["inci"] = sagittal[55]
    gui["sagittal"] = sagittal

    # Area function computation.
    alpha = ALPHA
    beta = BETA

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

    gui["B"] = lip_h * c
    gui["A"] = lip_w * c
    gui["area"] = af

    return gui


def showgui(gui):
    """Plot the sagittal contour of a VLAM state; returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # Constants used for the display adjustment.
    mx = -0.0153
    my = 138.4566

    # Extract the sagittal data and rescale.
    sag = gui["sagittal"] / 29.5

    h1, = ax.plot(
        (sag[:, 0] - sag[55, 0] + mx),
        (sag[:, 1] + sag[55, 1]) - my,
        "-b"
    )

    h1.set_color("g")
    h1.set_linewidth(2)

    ax.text(-0.5, -7.5, "VLAM Display/GIPSA-LAB", fontsize=18)

    ax.axis([-1.5, 8.5, -7.5, 2.5])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return fig

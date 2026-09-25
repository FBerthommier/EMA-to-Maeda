"""End-to-end synthesis pipeline: input text -> audio, formants, parameters.

Refactored from SynthsylShort/synthSYLVcourtes.py (synthsyl, synthwordfen and
the main() word/syllable loop).

``synthesize_text`` takes a string of words separated by spaces, each word
composed of syllables separated by dots (e.g. "ba.du abi"), and returns the
concatenated audio signal, the formant trajectories and the articulatory
parameter trajectories.
"""

import numpy as np

from synthesis import config
from synthesis.acoustic_model import freqevalNN, synthpause
from synthesis.lpc_synthesis import synthsimpleSYL, synthsimpleWORDfen
from synthesis.syllables import build_syllable
from synthesis.trajectories import arc, boucle, boucleplot, parse
from synthesis.vlam import initVLAMLength


def synthsyl(gui, Pval, syl, cf0, valrect):
    """Synthesize the audio of a single syllable and its formants."""
    f1, f2, f3 = [], [], []
    matspec = []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)

    f = np.array([f1, f2, f3])
    sig = synthsimpleSYL(np.array(matspec), syl["typ"].upper(), cf0, 1, 3)
    return sig, f


def synthwordfen(gui, Pval, word, cf0, valrect, dur):
    """Synthesize the audio of a multi-syllable word and its formants."""
    f1, f2, f3 = [], [], []
    matspec = []

    for k in range(len(Pval)):
        f1_k, f2_k, f3_k, gui, spec = freqevalNN(Pval[k, :], gui, valrect)
        f1.append(f1_k)
        f2.append(f2_k)
        f3.append(f3_k)
        matspec.append(spec)

    f = np.array([f1, f2, f3])
    sig = synthsimpleWORDfen(np.array(matspec), word, cf0, 1, 2, dur)
    return sig, f


def synthesize_text(Rs,
                    gui=None,
                    valrect=config.VALRECT,
                    dur=config.DUR,
                    cf0=config.CF0,
                    verbose=True):
    """Synthesize a space-separated sequence of dot-separated syllable words.

    Returns (sig, fval, Pv) : audio signal, formant trajectories (3 x T) and
    articulatory parameter trajectories (T x 7).
    """
    co = config.CO
    nu = config.NU
    K = config.K
    Kvoy = config.KVOY
    Kpause = config.KPAUSE
    Pexp = config.PEXP
    coefcen = config.COEFCEN
    tabvoy = config.VOWELS
    tabcons = config.CONSONANTS
    rho = config.RHO
    theta = config.THETA

    if gui is None:
        gui = initVLAMLength(config.INIT_PARAMS, config.INIT_TRACK_LENGTH)

    Cw = [word for word in Rs.split()]
    sig = []
    fval = np.array([])
    sylend = [0, 0]
    Pv = np.array([])

    for nword in range(len(Cw)):
        R = Cw[nword]

        C = [word for word in R.split(".")]
        nbsyl = len(C)

        booldeb, tabdeb, boolast, tablast = parse(C, tabvoy, tabcons, rho, theta)
        sylcheck = 1 if nbsyl > 1 else 0

        for nsyl in range(nbsyl):
            R = C[nsyl]

            # Departure vowel: previous syllable's last vowel, or default.
            if sylcheck and nsyl > 0:
                prev_boolast = boolast[nsyl - 1]
                curr_booldeb = booldeb[nsyl]
                prev_tablast = tablast[nsyl - 1]
                weight = (1 - prev_boolast) + coefcen * prev_boolast * curr_booldeb
                voydeb = [weight * prev_tablast[1], prev_tablast[0]]
            else:
                voydeb = [0.5, tabdeb[nsyl]]

            if verbose:
                from synthesis.syllables import syllable_types
                print(syllable_types(R, tabvoy, tabcons))

            syl, ok = build_syllable(R, tabvoy, tabcons, rho, theta, voydeb)
            if not ok:
                sylcheck = 0
                continue

            if sylcheck:
                Pval = boucle(syl, co, nu, K, Kvoy, dur, Pexp)
                Tval = boucleplot(syl, nu, K, Kvoy, dur, Pexp)

                if nsyl == 0:
                    word = ["O", syl["typ"]]
                    Pvalword = Pval
                    syldeb = syl["pt"][0]
                    sig1 = []
                    fval1 = []
                elif nsyl == nbsyl - 1:
                    sig1, fval1 = synthwordfen(gui, np.vstack([Pvalword, Pval]),
                                               word + [syl["typ"], "F"], cf0, valrect, dur)
                    Pvalpause = np.array(arc(np.vstack([sylend, syldeb]), co,
                                             np.arange(0, 7), dur, [-np.pi, 0], 0, nu, Kpause, Pexp))
                    sylend = syl["pt"][-1]
                    if Pv.size == 0:
                        Pv = np.vstack([Pvalpause, Pvalword, Pval])
                    else:
                        Pv = np.vstack([Pv, np.vstack([Pvalpause, Pvalword, Pval])])
                    sig = np.concatenate([sig, np.zeros(dur * config.SILENCE_SAMPLES_PER_FRAME), sig1])
                    if fval.size == 0:
                        fval = np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])
                    else:
                        fval = np.hstack([fval, np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])])
                else:
                    word.append(syl["typ"])
                    Pvalword = np.vstack([Pvalword, Pval])
                    sig1 = []
                    fval1 = []
            else:
                Pval = boucle(syl, co, nu, K, Kvoy, dur, Pexp)
                Tval = boucleplot(syl, nu, K, Kvoy, dur, Pexp)
                fig = plt_figure()
                plt_polar(fig, Tval, R)
                sig1, fval1 = synthsyl(gui, np.array(Pval), syl, cf0, valrect)
                Pvalpause = np.array(arc(np.vstack([sylend, syl["pt"][0]]), co,
                                         np.arange(0, 7), dur, [-np.pi, 0], 0, nu, Kpause, Pexp))
                sylend = syl["pt"][-1]

                sig = np.concatenate([sig, np.zeros(dur * config.SILENCE_SAMPLES_PER_FRAME), sig1])
                if fval.size == 0:
                    fval = np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])
                else:
                    fval = np.hstack([fval, np.hstack([synthpause(gui, Pvalpause, valrect).T, fval1])])
                if Pv.size == 0:
                    Pv = np.vstack([Pvalpause, Pval])
                else:
                    Pv = np.vstack([Pv, np.vstack([Pvalpause, Pval])])

    return np.array(sig), fval, Pv


def plt_figure():
    """Create (or reuse) the polar trajectory figure, like original figure 2."""
    import matplotlib.pyplot as plt
    from synthesis.plotting import close_on_enter

    fig = plt.figure(2)
    plt.clf()
    fig.canvas.mpl_connect("key_press_event", close_on_enter)
    return fig


def plt_polar(fig, Tval, title):
    """Draw the complex polar trajectory of the current syllable."""
    import matplotlib.pyplot as plt

    plt.polar(np.angle(Tval), np.abs(Tval))
    plt.polar(np.angle(Tval), np.abs(Tval), ".")
    plt.title(title, fontsize=22)

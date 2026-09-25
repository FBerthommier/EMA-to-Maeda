"""Constants and paths for the articulatory synthesizer.

Refactored from SynthsylShort/synthSYLVcourtes.py (constants that were
previously hard-coded inside main() and the synthesis functions).
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repository root is the parent directory of this package (Github/).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data directories (legacy locations: copyparam/, copyformants/, Maeda/,
# modelctw/, modelctwrandom/, paramodelctw/, paramodelctwrandom/, copywav/,
# wav/, mat/ at the repository root; this script itself does not read any of
# them, but the paths are kept here for consistency with the repository
# layout).
DATA_DIR = REPO_ROOT / "data"

# Default output directory for synthesized audio / parameters / videos.
RESULTS_DIR = REPO_ROOT / "results" / "synthesis"

# Default output file stems (kept identical to the original script, which
# wrote "essai.wav", "essai.npy" and "output.avi" in the working directory).
DEFAULT_AUDIO_STEM = "essai"
DEFAULT_VIDEO_NAME = "output.avi"

# ---------------------------------------------------------------------------
# Sampling rates and time constants
# ---------------------------------------------------------------------------
FS = 20000          # Audio sampling frequency (Hz)
FRAME_STEP = 2 * 5e-3   # Time step between successive spectral frames (s)
FRAME_STEP_VOWEL = 5e-3  # Step used by the (unused) single-vowel synthesizer
TRANSITION = 1e-3   # Inter-frame filter transition duration (s)

# Silence inserted between syllables, in samples (dur * 200 at 20 kHz).
SILENCE_SAMPLES_PER_FRAME = 200

# ---------------------------------------------------------------------------
# Synthesis defaults (main())
# ---------------------------------------------------------------------------
VALRECT = 0.75      # Soft-rectification coefficient applied to areas
PEXP = 2            # Exponent of the Tau-model arc profiles
DUR = 16            # Number of frames per arc segment
K = 1000            # Tau stiffness for consonant/transition arcs
KVOY = 1000         # Tau stiffness for vowel arcs
KPAUSE = 1000       # Tau stiffness for inter-syllable pauses
NU = -1             # Direction of arc traversal
CF0 = 1             # F0 scaling coefficient
COEFCEN = 0.5       # Weight of the final vowel target toward syllable center

# Growth/decay exponents of the syllable energy envelope.
ENV_ATTACK_EXP = 1  # cexp
ENV_DECAY_EXP = 3   # cdec

# ---------------------------------------------------------------------------
# Vocal tract physical constants (freqevalNN / spectrelec)
# ---------------------------------------------------------------------------
SPEED_OF_SOUND = 35100        # c    [cm/s, in model units]
AIR_DENSITY = 1.14e-3         # ro
WALL_LOSS_LAMBDA = 5.5e-5     # lambda_
HEAT_RATIO = 1.4              # eta
VISCOSITY_MU = 1.86e-4        # mu
SPECIFIC_HEAT_CP = 0.24       # cp
WALL_DAMPING_BP = 1600        # bp
WALL_MASS_MP = 1.4            # mp

# Formant computation grid.
NB_FREQ = 500
FMAX_SPEC = 10000

# ---------------------------------------------------------------------------
# Vowel / consonant inventory and targets (main())
# ---------------------------------------------------------------------------
# Vowel labels and their (theta, rho) polar targets on the Tau circle.
VOWELS = ["u", "o", "O", "a", "è", "é", "i", "y", "E"]
THETA = [np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi, 4 * np.pi / 3,
         3 * np.pi / 2, 5 * np.pi / 3, 5.5 * np.pi / 3, 5.5 * np.pi / 3]
RHO = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.7, 0.3]

CONSONANTS = ["b", "d", "g"]

# Default consonant targets [rho, theta] for b, d, g (first row is unused;
# indices 1 and 2 are overwritten at run time by consvalD / consvalG).
CONSONANT_TARGETS = np.array([[1.2, np.pi / 3],
                              [1.2, 0.0],
                              [1.1, 0.0]])

# Articulator sets used for each consonant position (rows: b, d, g;
# entries are 1-based parameter indices, 0 means unused).
ART = np.array([[1, 2, 6, 0],
                [1, 2, 3, 4],
                [1, 2, 3, 4]])

# Articulator sets used for double (geminate-like) consonant clusters.
ART1 = np.array([[1, 2, 3, 6],
                 [1, 2, 3, 1]])

# Base geometry of the Tau-model circle: [rho, offset, angle] rows.
CO = np.array([[-1.5, 0, np.pi],
               [-2.5, 0, -np.pi / 3],
               [3, 0, np.pi / 3],
               [-2.75, 0.5, np.pi],
               [3, 0, np.pi / 3],
               [2.5, 0.5, np.pi],
               [-2, 0, np.pi / 3]])

# Articulator labels for parameter trajectory plots.
PARAM_LABELS = ["J", "B", "D", "T", "LP", "LH", "Hy"]

# Initial VLAM speaker: neutral articulatory position, vocal tract length 195.
INIT_PARAMS = np.zeros(7)
INIT_TRACK_LENGTH = 195

# ---------------------------------------------------------------------------
# Playback flag (audio audition during synthesis; does not affect output data)
# ---------------------------------------------------------------------------
PLAYBACK_ENABLED = False

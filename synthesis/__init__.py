"""Articulatory synthesis package for short syllables (SYLV courtes).

Refactored from SynthsylShort/synthSYLVcourtes.py.

The package synthesizes vowel/consonant sequences (V, C, VV, CVC, ...) with:
- a Tau-model articulatory trajectory generator (polar coordinate arcs),
- the VLAM/Maeda-style sagittal articulatory model (converted from C to
  Matlab to Python at GIPSA-lab),
- an area-function -> frequency-response -> formant pipeline (lossless
  acoustic tube with Newton pole refinement),
- LPC-based source-filter audio synthesis,
plus optional video, GIF and plotting outputs.
"""

from synthesis import config  # noqa: F401

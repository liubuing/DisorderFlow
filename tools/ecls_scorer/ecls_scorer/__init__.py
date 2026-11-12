"""Standalone ECLS (epitope-conditioned likelihood shift) scorer.

ECLS(s) = H3 NLL(complex backbone) - H3 NLL(peptide-stripped Fab backbone),
scored with ProteinMPNN conditional log-likelihoods on user-supplied structures.

This package implements the published definition (see README) but does not
reproduce the frozen ECLS publication dataset end-to-end, and the frozen
temporal-final evaluation must never be rerun with it.
"""

__version__ = "0.1.0"

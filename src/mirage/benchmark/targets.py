"""Antigen / target sequences referenced by loaders."""

from __future__ import annotations

# UniProt P16422 (human EpCAM / TACSTD1), residues 24-265 — extracellular
# domain after signal-peptide cleavage. Matches the target SNAP folded
# against in its EpCAM VHH discrimination experiment (4MZV chain A).
EPCAM_ECD: str = (
    "QEECVCENYKLAVNCFVNNNRQCQCTSVGAQNTVICSKLAAKCLVMKAEMNGSKLG"
    "RRAKPEGALQNNDGLYDPDCDESGLFKAKQCNGTSTCWCVNTAGVRRTDKDTEITC"
    "SERVRTYWIIIELKHKAREKPYDSKSLRTALQKEITTRYQLDPKFITSILYENNVI"
    "TIDLVQNSSQKTQNDVDIADVAYYFEKDVKGESLFHSKKMDLTVNGEQLDLDPGQT"
    "LIYYVDEKAPEFSM"
)

# UniProt P05231 (human IL-6) signal peptide, residues 1-29. Present on the raw
# AVIDa-hIL6 antigen sequences (mature region starts at "VPPGEDSKD..."); absent
# from Champloo's PDB-derived antigens. Stripped during normalization so both
# datasets are featurized on the mature antigen.
IL6_SIGNAL_PEPTIDE: str = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP"

# Known precursor signal-peptide prefixes to strip from antigen sequences.
SIGNAL_PEPTIDES: tuple[str, ...] = (IL6_SIGNAL_PEPTIDE,)

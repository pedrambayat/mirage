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

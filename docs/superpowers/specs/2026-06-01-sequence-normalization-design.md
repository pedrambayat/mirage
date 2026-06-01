# Sequence normalization for the M-S gate — design

**Date:** 2026-06-01
**Status:** approved (design); implementation pending
**Branch:** `sequence-normalization`

## Problem

The Phase A M-S sequence-only gate is trained on Champloo (mature-domain
sequences from PDB) and applied, frozen, to the AVIDa-hIL6 orthogonal
real-negative set. The two datasets are **not preprocessed the same way**, so
the Tier-S bulk-composition features are computed on non-comparable inputs:

- **AVIDa VHHs** carry a ~22-residue N-terminal secretion **leader peptide**
  (`MKYLLPTAAAGLLLLAAQPAMA…`) that Champloo's PDB-derived VHHs lack. This
  inflates `binder_length` to ~150 vs Champloo's ~123 mean (≈ +3σ) and shifts
  the hydrophobic fraction.
- **His tags** appear on VHHs of *both* datasets, raggedly (e.g. Champloo
  `…TVSSHHHHHH`, AVIDa `…VTVSSHHHHHH`), and on some antigens.
- **AVIDa antigens** (IL-6 + 30 point mutants) carry the IL-6 **signal peptide**
  (29 aa, `MNSFSTSAFGPVAFSLGLLLVLPAAFPAP`) + a His6 tail; Champloo antigens are
  already mature (signal peptides cleaved before crystallization).

The first orthogonal run (`mirage_ms_orthogonal.json`) collapsed to all-negative
(recall 0, precision NaN). The collapse direction is partly an artifact of this
preprocessing mismatch — the VHH leader pushes AVIDa logits just below the
Champloo-frozen threshold. The M-S "no signal" conclusion (in-dist AUROC ≈ 0.36)
holds independently, but the orthogonal table is not a clean test until both
datasets are normalized the same way.

## Goal

One normalization rule, applied identically to every dataset entering the M-S
gate (Champloo in-dist, AVIDa, and labeled-EpCAM when it lands), reducing each
binder/antigen sequence to its comparable **mature domain** before featurization.
Then re-stage, re-freeze, and re-run, regenerating the committed Phase A
artifacts.

Non-goal: changing the M-S model, features, or the gate methodology. This is a
data-comparability fix, not a modeling change.

## Normalization recipe

New module `src/mirage/features/normalize.py`, two idempotent functions:

### `normalize_binder(seq: str) -> str`

ANARCI (IMGT scheme) numbers the sequence; take the aligned variable-domain span
(`query_start..query_end` from ANARCI's alignment details) and slice the input,
then **`rstrip` a trailing His run** to clean ANARCI's occasional 1-residue
overhang into the His tag. This removes the AVIDa leader, His tails on both
datasets, and framing residues uniformly, leaving the IMGT variable domain
(`QVQL…VTVSS`). Verified end-to-end: leader removed, His removed, clean `VTVSS`
terminus, 122-aa domain.

**HMMER dependency.** ANARCI is a thin wrapper over the HMMER `hmmscan` CLI,
which is **not** in mirage's uv env. It is provided by a **mirage-local** HMMER
built from source via `scripts/install_hmmer.sh` into `<repo>/.tools/hmmer/`
(gitignored). `normalize_binder` resolves the HMMER bin directory in order:
1. `MIRAGE_HMMER_BIN` env var,
2. `shutil.which("hmmscan")` (PATH),
3. `<repo>/.tools/hmmer/bin` (the bootstrap default).
This is deliberately **independent of the `mber` conda env** — mirage owns its
HMMER. ANARCI's germline HMM database ships pressed with the package
(`anarci/dat/HMMs/ALL.hmm.*`), so no further setup is needed.

**Fallback.** If ANARCI is unimportable or no HMMER bin is found or no Ig domain
is detected (e.g. a toy/non-VHH test string), return the His-stripped input
unchanged. Real staging on PARCC runs with HMMER present; the fallback keeps
HMMER-less environments (CI, other machines) from crashing, and is logged so
degraded staging is never silent.

### `normalize_antigen(seq: str) -> str`

Deterministic, dependency-free:
1. **Curated mature-start map** — if `seq` starts with a known precursor
   signal-peptide prefix, slice it off. v1 has one entry: human IL-6
   (`MNSFSTSAFGPVAFSLGLLLVLPAAFPAP…`, P05231 signal peptide residues 1–29 →
   mature starts at `VPPGEDSKD…`). All 31 AVIDa antigens share this prefix
   (mutations are in the mature region), so one rule covers them.
2. **His-tag strip** — remove an unambiguous terminal `H{5,}` block at either
   terminus (tolerating a short ≤3-residue Met/Ala cloning prefix and a short
   linker). Conservative: only cuts on clear His runs.

The IL-6 precursor/signal constant lives in `src/mirage/benchmark/targets.py`
(existing home for antigen sequence constants such as `EPCAM_ECD`).

Both functions are **idempotent** — re-applying is a no-op — which is what makes
the two-call-site wiring below safe.

## Wiring (Approach A — normalize at staging + harness, `sequence_features` stays pure)

- `scripts/stage_avida.py` — normalize `vhh_sequence` and `antigen_sequence`
  before writing the staged CSV. ANARCI memoized (`functools.lru_cache`) over
  *unique* sequences: AVIDa's 573,891 rows have only a few thousand unique VHHs
  and 31 antigens, so this is seconds, not hours.
- `scripts/stage_champloo_features.py` — normalize both sequences before
  `sequence_features(...)`.
- `src/mirage/eval/orthogonal.py::features_for_examples` — normalize each
  example's `binder_chains[0]` / `target_chains[0]` before featurizing. This is
  the guarantee that covers **labeled-EpCAM** (loader-only, no staging script)
  and is an idempotent no-op on the already-normalized staged AVIDa CSV.

`src/mirage/features/sequence.py` is unchanged and stays pure / dependency-light.

## Reporting robustness (folded in, write-boundary only)

Normalization may move AVIDa off the all-negative collapse, but if precision
remains undefined (0 predicted positives), the current code writes a bare `NaN`
token — invalid strict JSON. Scoped fix in `scripts/analyze_ms_orthogonal.py`
**only**: serialize undefined precision as `null` and make the stdout summary
line tolerate it. `metrics_at_threshold`'s contract (used by the in-dist
analysis and gate tests) is left untouched — minimal blast radius.

## Re-run sequence (regenerates committed artifacts — intended)

1. `bash scripts/install_hmmer.sh` — one-time HMMER bootstrap (done).
2. Implement `normalize.py` (+ targets constant) with tests.
3. Wire the three call sites; fix the orthogonal write boundary.
4. Re-stage AVIDa →
   `uv run python scripts/stage_avida.py --records data/raw/avida/AVIDa-hIL6.csv --antigens data/raw/avida/antigen_sequences.csv --output data/staged/avida/avida_staged.csv`
5. Re-stage Champloo features →
   `uv run python scripts/stage_champloo_features.py --pairs data/staged/champloo/champloo_pairs_af3.csv --supp ../abdisc-data/champloo/Supplementary_Table_1_*.csv --output data/staged/champloo/champloo_features_af3.csv`
6. Re-train / re-freeze →
   `uv run python scripts/analyze_ms_indist.py --features data/staged/champloo/champloo_features_af3.csv --model-out results/published/ms_model_af3.json --output results/published/mirage_ms_indist_af3.json`
   (defaults: l2=1.0, target-precision=0.9, seed=20260531)
7. Re-run orthogonal →
   `uv run python scripts/analyze_ms_orthogonal.py --model results/published/ms_model_af3.json --avida-csv data/staged/avida/avida_staged.csv --output results/published/mirage_ms_orthogonal.json`
8. Re-fill the in-dist + orthogonal rows of
   `results/published/mirage_phase_a_summary.md`.

labeled-EpCAM stays **blocked** (no `../abdisc-data/epcam/epcam_killing_labels.csv`)
and is documented as such — unchanged by this work.

## Testing & CI

- `tests/test_normalize.py`: leader+His VHH → variable domain; IL-6-with-signal
  antigen → mature; His-tail antigen → stripped; idempotency
  (`f(f(x)) == f(x)`); non-Ig fallback (toy string → His-stripped self). ANARCI
  tests **skip gracefully** when ANARCI/HMMER is unavailable, so CI
  (`uv sync → ruff → mypy → pytest`) stays green where HMMER is absent; the
  antigen / idempotency / fallback tests run dependency-free.
- Update existing `tests/test_avida_loader.py` staging tests to use
  already-mature toy inputs (pass-through), so they don't depend on ANARCI.
- Full battery before commit: `uv run ruff check`, `uv run ruff format --check`,
  `uv run mypy src/mirage`, `uv run pytest`.

## Risks / assumptions

- **Re-touching today's committed artifacts** (`ms_model_af3.json`,
  `mirage_ms_indist_af3.json`, `mirage_phase_a_summary.md`) is expected.
- **In-dist conclusion** (AUROC ≈ 0.36, "no signal") is very unlikely to change
  qualitatively — normalization removes a nuisance axis, it does not add binding
  signal. If it moves materially, that itself warrants a note.
- **HMMER coupling** is now mirage-local (built from source), not mber. The only
  residual coupling is the need to run the one-time bootstrap on each machine;
  documented in CLAUDE.md.
- **Work on branch `sequence-normalization`**; commits authored by Pedram, no
  Claude/Anthropic trailer (per workspace convention).

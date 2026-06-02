# M-C Phase B1 — Protenix Infrastructure & Feature Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Protenix shell-out predictor and a torch-free feature pipeline that turns predicted (VHH, antigen) complexes into a cached M-C feature CSV (confidence internals + interface geometry + CDR engagement) for the Champloo and SAbDab pair sets — the input substrate for the Phase B2 rung-ladder modeling.

**Architecture:** GPU work (Protenix) runs in a separate conda env via a SLURM array, exactly mirroring the existing `AF2MPosePredictor` three-phase contract (`stage → submit → results_for`). The mirage package stays torch-free: it only reads cached PDB/CIF + confidence JSON and emits feature rows. New feature extractors reuse the existing predictor-agnostic `StructuralInterfaceScorer` and the ANARCI normalization stack. The exact Protenix output schema is empirically captured in Task 1 and committed as a test fixture; all parsers are TDD'd against that fixture.

**Tech Stack:** Python 3.11, uv (mirage env, torch-free), numpy, BioPython, ANARCI + local HMMER, Protenix (separate conda env, B200 GPU via SLURM), pytest, ruff, mypy.

**Scope boundary:** This plan ends at cached feature CSVs + a validated cognate≫shuffled ipTM check. The rung-ladder trainer, paired-delta bootstrap, bidirectional cross-regime transfer, AF3 companion, and reporting are **Phase B2**, authored as a separate plan once these features exist (its column references depend on Task 6's output).

**Spec:** `docs/superpowers/specs/2026-06-02-mirage-mc-structure-track-design.md`.

**Branch:** create `mc-structure-track` off the open M-S **PR #2** branch (it carries the `eval/gate.py` random-split / contrast scaffolding Phase B2 needs), not off the partial `main`. If PR #2 is already merged, branch off `main`. Commits are **Pedram-authored, no Claude/Anthropic trailer**.

**Pre-flight (run once before Task 1):**
```bash
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
git fetch origin && git checkout -b mc-structure-track origin/sabdab-sequence-baseline  # PR #2 tip; fall back to origin/main if merged
uv sync && uv run pytest -q   # baseline green before adding anything
```

---

## File Structure

**New (mirage package, torch-free):**
- `src/mirage/scorers/protenix_confidence.py` — confidence-internals extractor (ipTM, pTM, interface-PAE, interface pLDDT, mean pLDDT) from a Protenix output dir.
- `src/mirage/pose_predictors/protenix.py` — `ProtenixPosePredictor` (stage/submit/results_for), templates-off baked in.
- `src/mirage/features/cdr_engagement.py` — ANARCI-IMGT CDR-vs-framework interface-contact fractions, row-preserving fallback.

**New (scripts):**
- `scripts/slurm/predict_protenix.slurm` + `scripts/slurm/run_protenix_chunk.py` — SLURM array wrapper + chunk runner (mirror the af2m pair).
- `scripts/stage_protenix_pairs.py` — build Champloo + SAbDab Protenix input manifests from a pairs CSV.
- `scripts/extract_mc_features.py` — assemble the staged M-C feature CSV from predicted complexes.

**New (docs / fixtures):**
- `docs/datasets/protenix-output-schema.md` — empirically captured output layout + JSON key names.
- `tests/fixtures/protenix/<id>/...` — a trimmed real Protenix output, committed as parser fixture.

**Modified:**
- `src/mirage/scorers/__init__.py` — side-effect import of `protenix_confidence`.
- `src/mirage/pose_predictors/__init__.py` — export `ProtenixPosePredictor`.
- `docs/datasets/dataset-registry.md` — note Protenix predictions now exist for Champloo + SAbDab.

**Reused unchanged:** `scorers/structural_interface.py`, `scorers/_structure.py`, `features/normalize.py` (HMMER resolver + ANARCI), `scripts/stage_sabdab_pairs.py` output, `pose_predictors/base.py`, `eval/gate.py`.

---

## Task 1: Protenix env install + single-pair smoke + schema capture (feasibility GATE)

This is an **operational spike**, not a TDD task: it proves Protenix runs on PARCC and captures the real output schema every later parser depends on. **Do not proceed past this task until the cognate pair returns a high ipTM and the schema doc + fixture are committed.**

**Files:**
- Create: `docs/datasets/protenix-output-schema.md`
- Create: `tests/fixtures/protenix/3OGO__3OGO/` (trimmed real output)

- [ ] **Step 1: Create the Protenix conda env (separate from mirage's uv env)**

```bash
conda create -y -n protenix python=3.11
conda activate protenix
pip install protenix            # ByteDance AF3-class predictor; pulls torch-CUDA
python -c "import protenix; print('protenix', getattr(protenix,'__version__','installed'))"
```
Expected: prints a version, no import error. (If the PyPI name differs in the installed version, install from the official Protenix GitHub release per its README — record the exact command in the schema doc.)

- [ ] **Step 2: Build a one-pair input from the Champloo cognate diagonal**

Use the cognate pair `3OGO` (antigen chain A + binder chain B). Pull the two sequences from the Champloo supplementary CSV:
```bash
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
uv run python - <<'PY'
import csv, json, pathlib
src = pathlib.Path("../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv")
rows = list(csv.DictReader(src.open()))
print("columns:", list(rows[0].keys()))   # identify the antigen-seq + vhh-seq + id columns
PY
```
Record which columns hold the antigen sequence, the VHH sequence, and the PDB id in `protenix-output-schema.md`. Write the chosen pair to a Protenix JSON input (per Protenix's input format from its README) with **two chains** (binder first, then antigen) and **templates disabled / MSA enabled**.

- [ ] **Step 3: Run Protenix on the single pair on a GPU**

```bash
srun -A dbgoodma-goodman-laboratory -p b200-mig45 --gres=gpu:1 --pty bash -lc '
  conda activate protenix
  # exact CLI per Protenix README; templates OFF, one seed, MSA on
  protenix predict --input data/raw/predictions/protenix/_smoke/3OGO.json \
                   --out_dir data/raw/predictions/protenix/_smoke/3OGO_out --seeds 0
'
```
Expected: completes without error and writes a structure (CIF/PDB) + confidence JSON(s).

- [ ] **Step 4: Document the output schema and capture a fixture**

Inspect the output directory. In `docs/datasets/protenix-output-schema.md` record, verbatim:
- the relative path of the top-ranked structure file (CIF/PDB) and its chain IDs;
- the confidence JSON filename(s) and the **exact keys** for: `iptm`, `ptm`, the PAE matrix, per-residue/per-atom pLDDT (and whether pLDDT is also in the structure B-factor column);
- the per-chain / chain-pair iptm fields if present.

Copy a **trimmed** version into `tests/fixtures/protenix/3OGO__3OGO/` (the confidence JSON as-is if small; if the PAE matrix is huge, downsample to a documented small NxN in the fixture and note it). This fixture is the parser's test oracle.

- [ ] **Step 5: Sanity-check + commit**

Confirm the cognate ipTM is high (a real binding pair should score well — e.g. ≳0.7; record the value). This is the first half of the §4 install-validation gate (the full cognate≫shuffled check lands in Task 7 once negatives run).
```bash
git add docs/datasets/protenix-output-schema.md tests/fixtures/protenix/
git commit -m "Document Protenix output schema + commit single-pair fixture (M-C feasibility gate)"
```

---

## Task 2: `protenix_confidence` extractor (TDD against the Task 1 fixture)

**Files:**
- Create: `src/mirage/scorers/protenix_confidence.py`
- Modify: `src/mirage/scorers/__init__.py`
- Test: `tests/test_protenix_confidence_scorer.py`

- [ ] **Step 1: Write the failing test (uses the committed fixture)**

```python
# tests/test_protenix_confidence_scorer.py
from pathlib import Path

from mirage.scorers.base import BenchmarkExample
from mirage.scorers.protenix_confidence import ProtenixConfidenceScorer

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "protenix"


def _example() -> BenchmarkExample:
    # binder chain first, antigen second — matches the staging convention
    return BenchmarkExample(
        id="3OGO__3OGO",
        binder_chains=("QVQLVESGGGLVQ",),   # placeholder seq; the scorer reads structure, not seq
        target_chains=("EVALPED",),
        label="1",
        binder_format="vhh",
    )


def test_emits_core_confidence_fields():
    scorer = ProtenixConfidenceScorer(predictions_root=FIXTURE_ROOT)
    score = scorer.score(_example())
    extras = score.extras
    for key in ("iptm", "ptm", "interface_pae", "interface_plddt", "mean_plddt"):
        assert key in extras
    assert 0.0 <= float(extras["iptm"]) <= 1.0
    assert float(score.value) == float(extras["iptm"])   # headline value == ipTM


def test_missing_prediction_is_nan_not_crash():
    scorer = ProtenixConfidenceScorer(predictions_root=FIXTURE_ROOT)
    ex = _example()
    ex = BenchmarkExample(id="does__not_exist", binder_chains=ex.binder_chains,
                          target_chains=ex.target_chains, label="1", binder_format="vhh")
    score = scorer.score(ex)
    assert score.extras.get("missing") == "prediction"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_protenix_confidence_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: mirage.scorers.protenix_confidence`.

- [ ] **Step 3: Implement the extractor**

Fill the `_FIELDS` constants from the Task 1 schema doc (this is the single, documented reconciliation point — not a placeholder). Interface residues reuse the 8 Å definition and `_structure` helpers already used by `StructuralInterfaceScorer`.

```python
# src/mirage/scorers/protenix_confidence.py
"""Predictor confidence internals from a Protenix output directory.

Reads ipTM / pTM (scalars), the PAE matrix, and per-residue pLDDT, and reduces
them to interface-localized confidence features. Confidence is predictor-specific
(unlike StructuralInterfaceScorer's geometry), so this scorer is Protenix-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage._paths import default_protenix_predictions_root
from mirage.scorers._registry import register
from mirage.scorers._structure import chain_residues, load_structure, predicted_chain_ids
from mirage.scorers.base import AbstractScorer, BenchmarkExample, Score

# --- confirm against docs/datasets/protenix-output-schema.md (Task 1) ---
_STRUCTURE_GLOB = "*model*.cif"          # top-ranked predicted structure
_CONFIDENCE_GLOB = "*summary_confidence*.json"
_PAE_GLOB = "*full_data*.json"           # file holding the PAE matrix
_KEY_IPTM = "iptm"
_KEY_PTM = "ptm"
_KEY_PAE = "pae"
_KEY_PLDDT = "atom_plddts"               # per-atom pLDDT; reduced to per-residue below
INTERFACE_CUTOFF_A = 8.0


@register("protenix_confidence")
class ProtenixConfidenceScorer(AbstractScorer):
    name = "protenix_confidence"

    def __init__(self, predictions_root: str | Path | None = None) -> None:
        if predictions_root is None:
            predictions_root = default_protenix_predictions_root()
        self.predictions_root = Path(predictions_root)

    def score(self, example: BenchmarkExample) -> Score:
        out_dir = self.predictions_root / example.id
        conf = next(out_dir.glob(_CONFIDENCE_GLOB), None)
        struct = next(out_dir.glob(_STRUCTURE_GLOB), None)
        if conf is None or struct is None:
            return self.nan_score(example, missing="prediction")
        try:
            return self._score_real(example, out_dir, conf, struct)
        except Exception as exc:  # never crash the batch
            return self.nan_score(example, error=f"{type(exc).__name__}: {exc}"[:200])

    def _score_real(self, example: BenchmarkExample, out_dir: Path,
                    conf_path: Path, struct_path: Path) -> Score:
        conf = json.loads(conf_path.read_text())
        iptm = float(conf[_KEY_IPTM])
        ptm = float(conf[_KEY_PTM])

        pred = load_structure(struct_path)
        binder_ids, target_ids = predicted_chain_ids(example)
        binder_res = [r for c in binder_ids for r in chain_residues(pred, c)]
        target_res = [r for c in target_ids for r in chain_residues(pred, c)]

        # interface residue masks via CB/heavy-atom 8 A contact
        inter_b, inter_t = _interface_masks(binder_res, target_res, INTERFACE_CUTOFF_A)

        pae = _load_pae(out_dir, conf)            # (N_res, N_res) over binder+target residues
        plddt_res = _per_residue_plddt(out_dir, conf, n_res=len(binder_res) + len(target_res))

        extras: dict[str, float | str] = {
            "iptm": iptm,
            "ptm": ptm,
            "interface_pae": _interface_pae(pae, len(binder_res), inter_b, inter_t),
            "interface_plddt": _interface_plddt(plddt_res, len(binder_res), inter_b, inter_t),
            "mean_plddt": float(np.nanmean(plddt_res)) if plddt_res.size else float("nan"),
        }
        return Score(example_id=example.id, scorer_name=self.name, value=iptm, extras=extras)
```

Add the small helpers `_interface_masks`, `_load_pae`, `_per_residue_plddt`, `_interface_pae`, `_interface_plddt` in the same file (interface mask = any cross-chain residue pair within 8 Å, computed with the same chunked heavy-atom distance approach as `StructuralInterfaceScorer._distance_stats`; `interface_pae` = mean of the binder-interface × antigen-interface PAE sub-block; `interface_plddt` = mean pLDDT over union of interface residues). Also add `default_protenix_predictions_root()` to `src/mirage/_paths.py` returning `<repo_root>/data/raw/predictions/protenix`.

- [ ] **Step 4: Register via side-effect import**

In `src/mirage/scorers/__init__.py`, add:
```python
import mirage.scorers.protenix_confidence  # noqa: F401
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_protenix_confidence_scorer.py -v`
Expected: PASS (both tests). Then `uv run ruff check && uv run mypy src/mirage`.

- [ ] **Step 6: Commit**

```bash
git add src/mirage/scorers/protenix_confidence.py src/mirage/scorers/__init__.py \
        src/mirage/_paths.py tests/test_protenix_confidence_scorer.py
git commit -m "Add Protenix confidence-internals scorer (iptm/ptm/interface-PAE/pLDDT)"
```

---

## Task 3: `ProtenixPosePredictor` wrapper + SLURM runner (TDD the pure-Python parts)

**Files:**
- Create: `src/mirage/pose_predictors/protenix.py`
- Create: `scripts/slurm/run_protenix_chunk.py`
- Create: `scripts/slurm/predict_protenix.slurm`
- Modify: `src/mirage/pose_predictors/__init__.py`
- Test: `tests/test_protenix_pose_predictor.py`

- [ ] **Step 1: Write the failing test (staging + sbatch command, no GPU)**

```python
# tests/test_protenix_pose_predictor.py
from mirage.pose_predictors.protenix import ProtenixPosePredictor
from mirage.scorers.base import BenchmarkExample


def _ex(i):
    return BenchmarkExample(id=f"p{i}", binder_chains=("QVQLVESGG",),
                            target_chains=("EVALPED",), label="1", binder_format="vhh")


def test_stage_writes_manifest_and_inputs(tmp_path):
    pred = ProtenixPosePredictor(output_root=tmp_path / "out", staged_root=tmp_path / "stage")
    manifest = pred.stage([_ex(0), _ex(1)])
    assert manifest.n_rows == 2
    assert manifest.path.is_file()
    header = manifest.path.read_text().splitlines()[0]
    assert header.split("\t") == ["example_id", "input_path", "out_dir"]


def test_stage_skips_cached(tmp_path):
    pred = ProtenixPosePredictor(output_root=tmp_path / "out", staged_root=tmp_path / "stage")
    (tmp_path / "out" / "p0").mkdir(parents=True)
    (tmp_path / "out" / "p0" / "rank1.pdb").write_text("X")
    manifest = pred.stage([_ex(0), _ex(1)])
    assert manifest.n_rows == 1 and manifest.n_already_cached == 1


def test_sbatch_command_has_templates_off_and_array(tmp_path):
    pred = ProtenixPosePredictor(output_root=tmp_path / "out", staged_root=tmp_path / "stage")
    manifest = pred.stage([_ex(0), _ex(1), _ex(2)])
    cmd = pred.sbatch_command(manifest, chunk_size=2, max_concurrent=4)
    assert "--array=0-1%4" in cmd          # ceil(3/2) = 2 tasks -> 0..1
    assert any("dbgoodma-goodman-laboratory" in c for c in cmd)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_protenix_pose_predictor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the predictor (mirror `af2m.py`)**

Create `src/mirage/pose_predictors/protenix.py` mirroring `AF2MPosePredictor` exactly, with these deltas: manifest columns `example_id, input_path, out_dir`; `stage()` writes a per-example Protenix JSON input (binder chain first, antigen second; templates disabled, MSA enabled) instead of a FASTA; `results_for` returns `<out>/<id>/rank1.pdb`; default partition `b200-mig45`, account `dbgoodma-goodman-laboratory`. Reuse the `_parse_sbatch_job_id`, `_clean_sequence`, chunk-array math from af2m. Add a `protenix_from_env()` factory and `default_protenix_predictions_root()` (already added in Task 2). Templates-off is a hard-coded input field, asserted in a test.

- [ ] **Step 4: Implement the SLURM runner + wrapper (mirror the af2m pair)**

`scripts/slurm/run_protenix_chunk.py`: stdlib-only; reads the manifest, slices its chunk, and for each row that lacks `rank1.pdb`, shells out to the Protenix CLI (templates off, seeds 0) into `out_dir`, then post-processes: symlink the top-ranked CIF/PDB to `rank1.pdb` and copy/normalize the confidence JSON to a stable filename. Mirror the structure of `run_af2m_chunk.py` (same logging, skip-cached, return-code semantics). `scripts/slurm/predict_protenix.slurm`: `#SBATCH` header with the account/partition/qos from the spec, `conda activate protenix`, then `python run_protenix_chunk.py "$1" "$SLURM_ARRAY_TASK_ID" "$2"`.

- [ ] **Step 5: Run tests + lint + types**

Run: `uv run pytest tests/test_protenix_pose_predictor.py -v && uv run ruff check && uv run mypy src/mirage`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/mirage/pose_predictors/protenix.py src/mirage/pose_predictors/__init__.py \
        scripts/slurm/run_protenix_chunk.py scripts/slurm/predict_protenix.slurm \
        tests/test_protenix_pose_predictor.py
git commit -m "Add ProtenixPosePredictor + SLURM runner (templates-off, MSA-on)"
```

---

## Task 4: `stage_protenix_pairs.py` — Champloo + SAbDab manifests (TDD)

**Files:**
- Create: `scripts/stage_protenix_pairs.py`
- Test: `tests/test_stage_protenix_pairs.py`

- [ ] **Step 1: Write the failing test (pairs-CSV → BenchmarkExamples → manifest)**

```python
# tests/test_stage_protenix_pairs.py
import csv
from pathlib import Path

from scripts.stage_protenix_pairs import examples_from_pairs_csv


def _write_pairs(p: Path):
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "binder_seq", "antigen_seq",
                                           "label", "antigen_cluster", "fold"])
        w.writeheader()
        w.writerow({"pair_id": "A__B", "binder_seq": "QVQ", "antigen_seq": "EVAL",
                    "label": "1", "antigen_cluster": "3", "fold": "0"})
        w.writerow({"pair_id": "A__B__neg0", "binder_seq": "QVQ", "antigen_seq": "WXYZ",
                    "label": "0", "antigen_cluster": "5", "fold": "0"})


def test_examples_from_pairs_csv(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    _write_pairs(csv_path)
    examples = list(examples_from_pairs_csv(csv_path))
    assert [e.id for e in examples] == ["A__B", "A__B__neg0"]
    assert examples[0].binder_chains == ("QVQ",)
    assert examples[0].target_chains == ("EVAL",)
    assert examples[0].label == "1"
    assert examples[1].label == "0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_stage_protenix_pairs.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the staging script**

```python
# scripts/stage_protenix_pairs.py
"""Stage Protenix inputs for a pairs CSV (Champloo or SAbDab).

Reads the flat pairs CSV emitted by stage_sabdab_pairs.py / the Champloo
equivalent (columns: pair_id, binder_seq, antigen_seq, label, antigen_cluster,
fold), turns each row into a BenchmarkExample (binder first, antigen second),
and runs ProtenixPosePredictor.stage() to write inputs + a manifest. Submits
when --submit is passed.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer

from mirage.pose_predictors.protenix import ProtenixPosePredictor
from mirage.scorers.base import BenchmarkExample

app = typer.Typer(add_completion=False)


def examples_from_pairs_csv(path: Path) -> Iterator[BenchmarkExample]:
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            yield BenchmarkExample(
                id=row["pair_id"],
                binder_chains=(row["binder_seq"],),
                target_chains=(row["antigen_seq"],),
                label=row["label"],
                binder_format="vhh",
                metadata={"antigen_cluster": row["antigen_cluster"], "fold": row["fold"]},
            )


@app.command()
def main(
    pairs: Annotated[Path, typer.Option(help="pairs CSV")],
    output_root: Annotated[Path, typer.Option(help="prediction output root")],
    staged_root: Annotated[Path, typer.Option(help="staged inputs root")],
    submit: Annotated[bool, typer.Option(help="submit the SLURM array")] = False,
    chunk_size: Annotated[int, typer.Option()] = 8,
    max_concurrent: Annotated[int, typer.Option()] = 16,
) -> None:
    pred = ProtenixPosePredictor(output_root=output_root, staged_root=staged_root)
    manifest = pred.stage(examples_from_pairs_csv(pairs))
    typer.echo(f"staged rows={manifest.n_rows} cached={manifest.n_already_cached}")
    if submit:
        job = pred.submit(manifest, chunk_size=chunk_size, max_concurrent=max_concurrent)
        typer.echo(f"submitted job={job}")


if __name__ == "__main__":
    app()
```

Confirm `BenchmarkExample` accepts a `metadata` kwarg (per `scorers/base.py`); if the field name differs, match it.

- [ ] **Step 4: Build the Champloo pairs CSV (reuse the SAbDab negative logic)**

Add a sibling helper `scripts/stage_champloo_protenix_pairs.py` OR a `--champloo-table` mode that reads `Supplementary_Table_1_final_experimental_vhh_ag_systems.csv`, takes the 91 cognate-diagonal positives, normalizes sequences, clusters antigens (reuse `features/clustering.py`), assigns folds, and constructs matched k=5 cross-cluster negatives by importing `build_pairs` from `scripts/stage_sabdab_pairs.py` (refactor `build_pairs` to accept a generic positives list if it is not already importable). Emit `data/staged/protenix/champloo_pairs.csv` with the **same six columns**. Add a unit test that the emitted CSV has 91 positives and 91*5 negatives.

- [ ] **Step 5: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_stage_protenix_pairs.py -v && uv run ruff check && uv run mypy src/mirage
git add scripts/stage_protenix_pairs.py scripts/stage_champloo_protenix_pairs.py \
        tests/test_stage_protenix_pairs.py
git commit -m "Add Protenix pair staging for Champloo + SAbDab (shared negative logic)"
```

---

## Task 5: CDR-engagement extractor (TDD) — `features/cdr_engagement.py`

**Files:**
- Create: `src/mirage/features/cdr_engagement.py`
- Test: `tests/test_cdr_engagement.py`

IMGT CDR position ranges (used to flag CDR residues): CDR1 = 27–38, CDR2 = 56–65, CDR3 = 105–117.

- [ ] **Step 1: Write the failing test (row-preserving fallback is the key contract)**

```python
# tests/test_cdr_engagement.py
import numpy as np

from mirage.features.cdr_engagement import cdr_engagement_features, CDR_FEATURE_NAMES


def test_fallback_is_row_preserving_on_anarci_failure():
    # An un-mappable "binder" (not an antibody) must yield defaults, not raise.
    feats = cdr_engagement_features(
        binder_seq="AAAAAAAAAA",
        binder_interface_residue_indices=np.array([0, 1, 2]),
        n_binder_residues=10,
    )
    assert feats["cdr_mapping_ok"] == 0.0
    assert feats["cdr_contact_fraction"] == 0.0
    assert set(CDR_FEATURE_NAMES).issubset(feats.keys())


def test_known_cdr_mapping_counts_interface_contacts():
    # A real VHH; residues that map to CDR positions should be flagged.
    vhh = ("QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKG"
           "RFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGLGTVVSEWDYDYWGQGTQVTVSS")
    n = len(vhh)
    # mark a spread of interface residues; with a real ANARCI mapping some land in CDRs
    feats = cdr_engagement_features(
        binder_seq=vhh,
        binder_interface_residue_indices=np.arange(0, n, 5),
        n_binder_residues=n,
    )
    assert feats["cdr_mapping_ok"] == 1.0
    assert 0.0 <= feats["cdr_contact_fraction"] <= 1.0
```
(The second test requires the local HMMER; mark it `@pytest.mark.skipif` on `_resolve_hmmer_bin() is None`, mirroring how other ANARCI-dependent tests guard.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cdr_engagement.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the extractor**

```python
# src/mirage/features/cdr_engagement.py
"""CDR-vs-framework engagement of a predicted antibody interface.

Maps IMGT CDR positions onto the binder chain via ANARCI, then reports what
fraction of the binder's *interface* residues fall in a CDR. Designed to be
ROW-PRESERVING: if ANARCI cannot map the chain (common for distorted loops in
non-cognate predicted poses), every feature defaults to 0.0 and `cdr_mapping_ok`
is 0.0 — the row is never dropped, so rung-to-rung deltas stay paired.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mirage.features.normalize import _resolve_hmmer_bin

# IMGT CDR position ranges (inclusive)
_CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
CDR_FEATURE_NAMES = (
    "cdr_contact_fraction",
    "cdr1_contact_fraction",
    "cdr2_contact_fraction",
    "cdr3_contact_fraction",
    "cdr_mapping_ok",
)


def _defaults() -> dict[str, float]:
    return {name: 0.0 for name in CDR_FEATURE_NAMES}


def _imgt_cdr_mask(binder_seq: str, n_residues: int) -> np.ndarray[Any, Any] | None:
    """Per-residue dict of CDR membership for the binder chain, aligned to the
    predicted chain's residue order. Returns None if ANARCI/HMMER is unavailable
    or the chain does not map to an antibody domain."""
    hmmer_bin = _resolve_hmmer_bin()
    if hmmer_bin is None:
        return None
    try:
        from anarci import run_anarci  # type: ignore[import-untyped]

        numbering, details, _ = run_anarci(
            [("q", binder_seq)], scheme="imgt", hmmerpath=hmmer_bin
        )[1:4] if False else (None, None, None)  # placeholder; see real call below
    except Exception:
        return None
    return None  # replaced in real implementation
```

Then implement `_imgt_cdr_mask` properly: call `run_anarci([("q", binder_seq)], scheme="imgt", hmmerpath=hmmer_bin)`, read `result[1][0]` (the numbering) and `result[2][0]` (domain details with `query_start`). Walk the numbered residues, and for each that is not a gap, record its sequence index (offset by `query_start`) and whether its IMGT number falls in any `_CDR_RANGES`. Build a boolean array of length `n_residues` (`True` where a residue index is a CDR position). Return it, or `None` on any failure.

Finally the public function:
```python
def cdr_engagement_features(
    binder_seq: str,
    binder_interface_residue_indices: np.ndarray[Any, Any],
    n_binder_residues: int,
) -> dict[str, float]:
    mask = _imgt_cdr_mask(binder_seq, n_binder_residues)
    if mask is None or binder_interface_residue_indices.size == 0:
        return _defaults()
    iface = binder_interface_residue_indices[binder_interface_residue_indices < mask.size]
    if iface.size == 0:
        return _defaults()
    cdr_hits = mask[iface]
    feats = {"cdr_contact_fraction": float(cdr_hits.mean()), "cdr_mapping_ok": 1.0}
    # per-CDR fractions need per-CDR masks; build them the same way in _imgt_cdr_mask
    # and pass through (see note). For now compute from a per-CDR mask dict.
    return feats
```
Adjust `_imgt_cdr_mask` to return a dict of four boolean arrays (`any`, `cdr1`, `cdr2`, `cdr3`) so the per-CDR fractions are real (not placeholders); the public function then fills all five `CDR_FEATURE_NAMES`. Remove the scaffolding `if False` line — that was only to show the call shape.

- [ ] **Step 4: Run tests (with HMMER) + lint + types**

```bash
bash scripts/install_hmmer.sh   # if .tools/hmmer not present
uv run pytest tests/test_cdr_engagement.py -v && uv run ruff check && uv run mypy src/mirage
```
Expected: PASS (fallback test always; mapping test if HMMER present).

- [ ] **Step 5: Commit**

```bash
git add src/mirage/features/cdr_engagement.py tests/test_cdr_engagement.py
git commit -m "Add CDR-engagement interface features with row-preserving ANARCI fallback"
```

---

## Task 6: `extract_mc_features.py` — assemble the staged M-C feature CSV (TDD)

**Files:**
- Create: `scripts/extract_mc_features.py`
- Test: `tests/test_extract_mc_features.py`

Curated geometry subset (from `StructuralInterfaceScorer.extras`): `n_interface_residues_binder`, `n_interface_residues_target`, `buried_sasa_proxy_a2`, `atom_contacts_5a`, `shape_complementarity_proxy`, `atom_clash_fraction_2a`. Confidence subset (Task 2): `iptm`, `ptm`, `interface_pae`, `interface_plddt`, `mean_plddt`. CDR subset (Task 5): the five `CDR_FEATURE_NAMES`. Plus passthrough: `pair_id`, `label`, `antigen_cluster`, `fold`.

- [ ] **Step 1: Write the failing test (assembly + missing-prediction handling)**

```python
# tests/test_extract_mc_features.py
import csv
from pathlib import Path

import pytest

from scripts.extract_mc_features import FEATURE_COLUMNS, assemble_row


def test_feature_columns_are_stable_and_complete():
    for col in ("pair_id", "label", "antigen_cluster", "fold",
                "iptm", "interface_pae", "n_interface_residues_binder",
                "cdr_contact_fraction", "cdr_mapping_ok"):
        assert col in FEATURE_COLUMNS


def test_assemble_row_marks_missing_prediction(tmp_path):
    row = {"pair_id": "X__Y", "binder_seq": "QVQ", "antigen_seq": "EVAL",
           "label": "1", "antigen_cluster": "3", "fold": "0"}
    out = assemble_row(row, predictions_root=tmp_path)  # no prediction on disk
    assert out["pair_id"] == "X__Y"
    assert out["prediction_present"] == "0"
    assert out["iptm"] == ""   # blank, not a fabricated number
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_extract_mc_features.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the assembler**

Compose the three extractors per pair: instantiate `StructuralInterfaceScorer(predictions_root=...)` and `ProtenixConfidenceScorer(predictions_root=...)`, score the `BenchmarkExample`, and call `cdr_engagement_features` using the binder interface-residue indices from the structural scorer (expose them: have the structural scorer's interface computation reused, or recompute the 8 Å binder-interface mask in the assembler from the predicted PDB via `_structure` helpers — pick one and keep it DRY by importing the helper). Write `FEATURE_COLUMNS` (passthrough + confidence + geometry + CDR + `prediction_present`). `assemble_row` returns blanks (`""`) for feature columns when the prediction is absent, with `prediction_present="0"` — never fabricate values. The CLI iterates a pairs CSV and writes `data/staged/mc/<dataset>_features.csv`.

- [ ] **Step 4: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_extract_mc_features.py -v && uv run ruff check && uv run mypy src/mirage
git add scripts/extract_mc_features.py tests/test_extract_mc_features.py
git commit -m "Add M-C feature assembler (confidence + interface geometry + CDR)"
```

---

## Task 7: Run the campaign + cognate≫shuffled validation (operational; B1→B2 boundary)

This task produces the cached feature CSVs. It is operational (SLURM + GPU), so steps are commands + acceptance criteria, not unit tests.

- [ ] **Step 1: Stage + submit the SAbDab campaign (the apples-to-apples rows)**

```bash
uv run python scripts/stage_protenix_pairs.py \
  --pairs data/staged/sabdab/sabdab_pairs.csv \
  --output-root data/raw/predictions/protenix/sabdab \
  --staged-root data/staged/protenix/sabdab --submit --chunk-size 8 --max-concurrent 16
```
Expected: `staged rows=2688 cached=0` then a submitted job id. Monitor with `squeue -u $USER`.

- [ ] **Step 2: Build + submit the Champloo campaign**

```bash
uv run python scripts/stage_champloo_protenix_pairs.py \
  --champloo-table ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \
  --output data/staged/protenix/champloo_pairs.csv
uv run python scripts/stage_protenix_pairs.py \
  --pairs data/staged/protenix/champloo_pairs.csv \
  --output-root data/raw/predictions/protenix/champloo \
  --staged-root data/staged/protenix/champloo --submit --chunk-size 8 --max-concurrent 16
```
Expected: `staged rows=546 cached=0` then a job id.

- [ ] **Step 3: Wait for completion + check coverage**

Poll until both arrays finish. Then verify prediction coverage:
```bash
uv run python - <<'PY'
from pathlib import Path
for name, n in [("sabdab", 2688), ("champloo", 546)]:
    root = Path(f"data/raw/predictions/protenix/{name}")
    done = sum(1 for d in root.iterdir() if (d / "rank1.pdb").is_file()) if root.exists() else 0
    print(name, done, "/", n)
PY
```
Acceptance: ≥ 99% of rows have `rank1.pdb`. Re-submit (idempotent stage skips cached) for any stragglers; record any permanent failures.

- [ ] **Step 4: §4 install-validation GATE — cognate ≫ shuffled ipTM on the Champloo diagonal**

```bash
uv run python - <<'PY'
import csv, statistics
from pathlib import Path
from mirage.scorers.protenix_confidence import ProtenixConfidenceScorer
from mirage.scorers.base import BenchmarkExample
root = Path("data/raw/predictions/protenix/champloo")
sc = ProtenixConfidenceScorer(predictions_root=root)
pos, neg = [], []
for r in csv.DictReader(open("data/staged/protenix/champloo_pairs.csv")):
    ex = BenchmarkExample(id=r["pair_id"], binder_chains=(r["binder_seq"],),
                          target_chains=(r["antigen_seq"],), label=r["label"], binder_format="vhh")
    v = sc.score(ex).extras.get("iptm")
    if isinstance(v, float):
        (pos if r["label"] == "1" else neg).append(v)
print("cognate ipTM median", statistics.median(pos))
print("shuffled ipTM median", statistics.median(neg))
PY
```
Acceptance: cognate median ipTM clearly above shuffled median (the predictor distinguishes real pairs at all). If they overlap, STOP and investigate the install/inputs before B2 — the predictor is not usable.

- [ ] **Step 5: Materialize the cached M-C feature CSVs**

```bash
uv run python scripts/extract_mc_features.py \
  --pairs data/staged/sabdab/sabdab_pairs.csv \
  --predictions-root data/raw/predictions/protenix/sabdab \
  --output data/staged/mc/sabdab_features.csv
uv run python scripts/extract_mc_features.py \
  --pairs data/staged/protenix/champloo_pairs.csv \
  --predictions-root data/raw/predictions/protenix/champloo \
  --output data/staged/mc/champloo_features.csv
```
Acceptance: both CSVs have one row per pair, the full `FEATURE_COLUMNS` header, `prediction_present=1` for ≥99% of rows, and a sane CDR-mapping failure rate (record it — high failure on negatives is itself a finding per the spec).

- [ ] **Step 6: Document + commit the run record (not the large prediction blobs)**

Update `docs/datasets/dataset-registry.md` (Protenix predictions now exist for Champloo + SAbDab; note counts + CDR-mapping failure rate + the cognate/shuffled ipTM medians). The `data/raw/predictions/` and `data/staged/` trees are gitignored — commit only the doc + a small `results/published/mc_campaign_record.md` capturing job ids, coverage, the gate medians, and failure rates.
```bash
git add docs/datasets/dataset-registry.md results/published/mc_campaign_record.md
git commit -m "Record M-C Protenix campaign: coverage, cognate/shuffled ipTM gate, CDR failure rate"
```

---

## Phase B2 preview (separate plan, authored after Task 7)

Once `data/staged/mc/{sabdab,champloo}_features.csv` exist, Phase B2 will add: a `paired_delta_bootstrap` helper in `eval/gate.py` (ΔAUROC / Δprecision on shared resampled rows, stratified like `bootstrap_ci`); a rung-ladder trainer that calls `train_ms` over the four feature-column subsets (rung 0 = `[iptm]`, rung 1 = + confidence internals, rung 2 = + geometry subset, rung 3 = + CDR); `analyze_mc_indist.py` (SAbDab OOF on the same folds, floor-check vs 0.496, paired rung-deltas, `--random-folds` contrast, SHAP); `analyze_mc_cross_regime.py` (bidirectional 2a SAbDab→Champloo / 2b Champloo→SAbDab with antigen dedup guard, via the `FrozenGate` harness); `analyze_mc_af3_companion.py` (Champloo AF3 rung-0→2 ladder + AF3-vs-Protenix ipTM comparison); and `results/published/mc_structure_summary.md`. Its exact column names come from Task 6's `FEATURE_COLUMNS`, which is why it is deferred to its own plan.

---

## Self-Review

**Spec coverage:** Protenix predictor + templates-off (Task 3) ✓; single-predictor-across-both campaign (Task 7) ✓; predict-the-shuffled-pair negatives, full 2,688 SAbDab + 546 Champloo (Tasks 4, 7) ✓; rung-0→3 feature substrate — confidence internals (Task 2), geometry subset (Task 6 reusing StructuralInterfaceScorer), CDR engagement with row-preserving fallback (Task 5) ✓; install-validation cognate≫shuffled gate (Tasks 1 + 7) ✓; torch-free invariant (all GPU shelled out) ✓; MSA caching note is operational (Protenix-side, recorded in schema doc). Modeling/eval items (paired deltas, bidirectional transfer, AF3 companion, random-split, SHAP, 0.496 floor-check) are explicitly **Phase B2** — covered by the preview, not this plan.

**Placeholder scan:** The only deliberate empirical-reconciliation point is the Protenix `_FIELDS` constants in Task 2, resolved by the Task 1 schema doc + committed fixture (not a silent TODO). The `if False` scaffolding line in Task 5 Step 3 is explicitly flagged for removal and replaced by the described real `run_anarci` call.

**Type consistency:** `BenchmarkExample` fields (`id`, `binder_chains`, `target_chains`, `label`, `binder_format`, `metadata`) match `scorers/base.py`; `StagedManifest` columns differ from af2m by design (`input_path` not `fasta_path`) and the test asserts the new header; `CDR_FEATURE_NAMES` is defined once (Task 5) and consumed in Task 6; `default_protenix_predictions_root` is created in Task 2 and reused in Task 3.

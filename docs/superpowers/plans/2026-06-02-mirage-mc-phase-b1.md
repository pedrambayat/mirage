# M-C Phase B1 — Protenix Infrastructure & Feature Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Protenix shell-out predictor and a torch-free feature pipeline that turns predicted (VHH, antigen) complexes into a cached M-C feature CSV (confidence internals + interface geometry + CDR engagement) for the Champloo and SAbDab pair sets — the input substrate for the Phase B2 rung-ladder modeling.

**Architecture:** GPU work (Protenix) runs in a separate conda env via a SLURM array, mirroring the existing `AF2MPosePredictor` three-phase contract (`stage → submit → results_for`). The mirage package stays torch-free: it only reads cached structure + confidence JSON and emits feature rows. **Three hardening invariants, baked in from the start:** (1) MSAs are computed **once per unique sequence** in a dedicated pre-compute step and reused across all pairings — per-pair jobs never run an MSA search; (2) binder/target chain roles are resolved by **sequence alignment to the predicted structure**, never by positional chain order (Protenix may reorder chains); (3) feature assembly is **fault-tolerant** — a missing/truncated prediction logs to `failed_pairs.txt` and is skipped, never crashing the batch. New feature extractors reuse the predictor-agnostic `StructuralInterfaceScorer` and the ANARCI normalization stack. The exact Protenix output + input schema is empirically captured in Task 1 and committed as a test fixture; all parsers are TDD'd against it.

**Tech Stack:** Python 3.11, uv (mirage env, torch-free), numpy, BioPython, ANARCI + local HMMER, Protenix (separate conda env, B200 GPU via SLURM), pytest, ruff, mypy.

**Scope boundary:** This plan ends at cached feature CSVs + a validated cognate≫shuffled ipTM check. The rung-ladder trainer, paired-delta bootstrap, bidirectional cross-regime transfer, AF3 companion, and reporting are **Phase B2**, authored as a separate plan once these features exist (its column references depend on Task 8's output).

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
- `src/mirage/scorers/_structure.py` — **modify**: add `resolve_chain_roles_by_sequence()` (sequence→chain-role mapping) and `read_chain_roles_json()`.
- `src/mirage/scorers/structural_interface.py` — **modify**: factor feature computation into `score_with_chains(example, pred_path, binder_ids, target_ids)`; `score()` gains a `chain_resolution` mode (`"positional"` default = AF2-M back-compat; `"sequence"` for Protenix).
- `src/mirage/scorers/protenix_confidence.py` — **new**: confidence-internals extractor (ipTM, pTM, interface-PAE, interface pLDDT, mean pLDDT), sequence-resolved chains.
- `src/mirage/pose_predictors/protenix.py` — **new**: `ProtenixPosePredictor` (stage/submit/results_for), templates-off + precomputed-MSA input.
- `src/mirage/features/cdr_engagement.py` — **new**: ANARCI-IMGT CDR-vs-framework interface-contact fractions, row-preserving fallback.

**New (scripts):**
- `scripts/slurm/predict_protenix.slurm` + `scripts/slurm/run_protenix_chunk.py` — SLURM array wrapper + chunk runner (precomputed-MSA mode; writes `chain_roles.json` in post-process).
- `scripts/slurm/precompute_msa.slurm` + `scripts/precompute_protenix_msa.py` — dedup unique sequences → run MSA once each → `.a3m` cache keyed by sequence hash.
- `scripts/stage_protenix_pairs.py` — per-pair Protenix JSON (referencing cached MSAs) + the deduplicated unique-sequence manifest.
- `scripts/stage_champloo_protenix_pairs.py` — Champloo 91 positives + matched k=5 negatives → pairs CSV.
- `scripts/extract_mc_features.py` — fault-tolerant feature assembler (sequence-resolved chains).

**New (docs / fixtures):**
- `docs/datasets/protenix-output-schema.md` — captured input+output layout, JSON keys, **precomputed-MSA field**, **observed output chain order**.
- `tests/fixtures/protenix/<id>/...` — a trimmed real Protenix output, committed as parser fixture.

**Modified:** `src/mirage/scorers/__init__.py` (side-effect import), `src/mirage/pose_predictors/__init__.py` (export), `docs/datasets/dataset-registry.md`.

**Reused unchanged:** `scorers/_structure.py` distance/interface helpers, `features/normalize.py` (HMMER resolver + ANARCI), `scripts/stage_sabdab_pairs.py` output, `pose_predictors/base.py`, `eval/gate.py`.

---

## Task 1: Protenix env install + single-pair smoke + schema capture (feasibility GATE)

Operational spike, not TDD: prove Protenix runs on PARCC and capture the real input+output schema every later step depends on. **Do not proceed until: cognate ipTM is high, AND the schema doc records (a) the precomputed-MSA input field and (b) the output chain IDs/order, AND the fixture is committed.**

**Files:** Create `docs/datasets/protenix-output-schema.md`, `tests/fixtures/protenix/3OGO__3OGO/`.

- [ ] **Step 1: Create the Protenix conda env**

```bash
conda create -y -n protenix python=3.11
conda activate protenix
pip install protenix
python -c "import protenix; print('protenix', getattr(protenix,'__version__','installed'))"
```
Expected: prints a version. If the PyPI name differs, install from the official Protenix GitHub release per its README and record the exact command in the schema doc.

- [ ] **Step 2: Build a one-pair input from the Champloo cognate diagonal**

```bash
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
uv run python - <<'PY'
import csv, pathlib
src = pathlib.Path("../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv")
rows = list(csv.DictReader(src.open()))
print("columns:", list(rows[0].keys()))   # identify antigen-seq, vhh-seq, pdb-id columns
PY
```
Record which columns hold antigen seq / VHH seq / PDB id in the schema doc. Write `3OGO` as a Protenix JSON input, two chains (binder first, antigen second), **templates disabled, MSA enabled**.

- [ ] **Step 3: Run Protenix on the single pair on a GPU**

```bash
srun -A dbgoodma-goodman-laboratory -p b200-mig45 --gres=gpu:1 --pty bash -lc '
  conda activate protenix
  protenix predict --input data/raw/predictions/protenix/_smoke/3OGO.json \
                   --out_dir data/raw/predictions/protenix/_smoke/3OGO_out --seeds 0
'   # exact CLI per Protenix README
```
Expected: completes; writes a structure (CIF/PDB) + confidence JSON(s).

- [ ] **Step 4: Document the schema (input + output) and capture a fixture**

In `docs/datasets/protenix-output-schema.md` record verbatim:
- **Output structure** file path, format, and the **chain IDs assigned to binder vs antigen** — explicitly note whether Protenix preserved the input order (binder-first) or reordered. This decides nothing in code (we resolve by sequence) but documents the reorder risk.
- **Confidence JSON** filename(s) and exact keys for: `iptm`, `ptm`, the PAE matrix, per-residue/per-atom pLDDT (and whether pLDDT is in the structure B-factor).
- **Precomputed-MSA input field** — how the input JSON references a precomputed `.a3m` / MSA dir per chain (e.g. a `"msa"`/`"precomputed_msa_dir"` field), and the CLI flag to **disable** the built-in MSA search (e.g. `--use_msa_server false`). This is what Task 5/6 wire up.

Copy a trimmed output into `tests/fixtures/protenix/3OGO__3OGO/` (confidence JSON as-is if small; downsample a huge PAE matrix to a documented small NxN). This fixture is the parser oracle.

- [ ] **Step 5: Sanity-check + commit**

Confirm cognate ipTM is high (≳0.7; record it). First half of the §4 install gate.
```bash
git add docs/datasets/protenix-output-schema.md tests/fixtures/protenix/
git commit -m "Document Protenix input+output schema (MSA field, chain order) + single-pair fixture"
```

---

## Task 2: Sequence-based chain-role resolver + StructuralInterfaceScorer chain injection

Fixes the chain-reordering trap at its root: resolve binder/target by **sequence**, and let the geometry scorer accept explicit chain IDs. `predicted_chain_ids` (positional) stays the default so AF2-M is untouched.

**Files:** Modify `src/mirage/scorers/_structure.py`, `src/mirage/scorers/structural_interface.py`. Test `tests/test_chain_role_resolver.py`.

- [ ] **Step 1: Write the failing test (reordered chains must still resolve correctly)**

```python
# tests/test_chain_role_resolver.py
from pathlib import Path

from mirage.scorers._structure import load_structure, resolve_chain_roles_by_sequence

# A 2-chain PDB where chain "A" is the ANTIGEN and chain "B" is the BINDER
# (i.e. NOT the positional binder-first assumption). Tiny: a few CA-only residues.
PDB = """\
ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  PRO A   3       7.600   0.000   0.000  1.00  0.00           C
ATOM      4  CA  GLN B   1       0.000  10.000   0.000  1.00  0.00           C
ATOM      5  CA  VAL B   2       3.800  10.000   0.000  1.00  0.00           C
ATOM      6  CA  TRP B   3       7.600  10.000   0.000  1.00  0.00           C
END
"""


def test_resolves_roles_by_sequence_not_position(tmp_path: Path):
    p = tmp_path / "x.pdb"
    p.write_text(PDB)
    struct = load_structure(p)
    binder_ids, target_ids = resolve_chain_roles_by_sequence(
        struct, binder_seqs=("QVW",), target_seqs=("GAP",)
    )
    assert binder_ids == ["B"]   # binder seq QVW lives on chain B
    assert target_ids == ["A"]   # antigen seq GAP lives on chain A
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_chain_role_resolver.py -v`
Expected: FAIL (`ImportError: cannot import name 'resolve_chain_roles_by_sequence'`).

- [ ] **Step 3: Implement the resolver in `_structure.py`**

```python
# add to src/mirage/scorers/_structure.py
from Bio.PDB.Polypeptide import three_to_one  # at top with other imports

def _chain_one_letter_seq(structure: Structure, chain_id: str) -> str:
    out = []
    for res in chain_residues(structure, chain_id):
        try:
            out.append(three_to_one(res.get_resname()))
        except KeyError:
            out.append("X")
    return "".join(out)


def _identity(a: str, b: str) -> float:
    """Fraction of the shorter sequence matched when one contains the other's core.
    Predicted chains carry the exact submitted sequence (predictors don't mutate
    sequence), so a real match is ~1.0; unmodeled termini only shorten it."""
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short in long:
        return 1.0
    m = sum(1 for x, y in zip(short, long) if x == y)
    return m / len(short)


def resolve_chain_roles_by_sequence(
    structure: Structure,
    binder_seqs: tuple[str, ...],
    target_seqs: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Map biological role -> output chain IDs by matching each output chain's
    sequence to the submitted binder/target sequences. Greedy best-identity
    assignment; each output chain is claimed at most once. Robust to predictors
    that reorder chains relative to the input."""
    chain_ids = [c.id for c in structure[0]]
    chain_seqs = {cid: _chain_one_letter_seq(structure, cid) for cid in chain_ids}
    used: set[str] = set()

    def _claim(seq: str) -> str:
        best_id, best_score = "", -1.0
        for cid in chain_ids:
            if cid in used:
                continue
            s = _identity(seq, chain_seqs[cid])
            if s > best_score:
                best_id, best_score = cid, s
        if best_id:
            used.add(best_id)
        return best_id

    binder_ids = [cid for seq in binder_seqs if (cid := _claim(seq))]
    target_ids = [cid for seq in target_seqs if (cid := _claim(seq))]
    return binder_ids, target_ids


def read_chain_roles_json(path: Path) -> tuple[list[str], list[str]] | None:
    """Read a {"binder": [...], "target": [...]} sidecar written at predict time."""
    if not path.is_file():
        return None
    import json
    data = json.loads(path.read_text())
    return list(data["binder"]), list(data["target"])
```

- [ ] **Step 4: Refactor `StructuralInterfaceScorer` to accept explicit chains**

In `structural_interface.py`: rename the body of `_score_real` to `score_with_chains(self, example, pred_path, binder_ids, target_ids)` (it currently calls `predicted_chain_ids(example)` at the top — replace that call with the passed-in `binder_ids, target_ids`). Add a constructor arg `chain_resolution: str = "positional"`. New `_score_real` resolves chains then delegates:

```python
def _score_real(self, example, pred_path):
    pred = load_structure(pred_path)
    if self.chain_resolution == "sequence":
        roles = read_chain_roles_json(pred_path.parent / "chain_roles.json")
        binder_ids, target_ids = roles or resolve_chain_roles_by_sequence(
            pred, example.binder_chains, example.target_chains)
    else:
        binder_ids, target_ids = predicted_chain_ids(example)
    return self.score_with_chains(example, pred, binder_ids, target_ids)
```
(Adjust `score_with_chains` to take the already-loaded `pred` rather than re-loading.) Add a regression test that `chain_resolution="positional"` on an AF2-M-style fixture yields identical extras to before.

- [ ] **Step 5: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_chain_role_resolver.py tests/test_structural_interface_scorer.py -v
uv run ruff check && uv run mypy src/mirage
git add src/mirage/scorers/_structure.py src/mirage/scorers/structural_interface.py \
        tests/test_chain_role_resolver.py
git commit -m "Resolve binder/target chains by sequence; inject explicit chains into geometry scorer"
```

---

## Task 3: `protenix_confidence` extractor (TDD against the Task 1 fixture)

**Files:** Create `src/mirage/scorers/protenix_confidence.py`; modify `src/mirage/scorers/__init__.py`, `src/mirage/_paths.py`. Test `tests/test_protenix_confidence_scorer.py`.

- [ ] **Step 1: Write the failing test (uses the committed fixture)**

```python
# tests/test_protenix_confidence_scorer.py
from pathlib import Path

from mirage.scorers.base import BenchmarkExample
from mirage.scorers.protenix_confidence import ProtenixConfidenceScorer

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "protenix"


def _example() -> BenchmarkExample:
    return BenchmarkExample(id="3OGO__3OGO", binder_chains=("QVQLVESGGGLVQ",),
                            target_chains=("EVALPED",), label="1", binder_format="vhh")


def test_emits_core_confidence_fields():
    scorer = ProtenixConfidenceScorer(predictions_root=FIXTURE_ROOT)
    score = scorer.score(_example())
    for key in ("iptm", "ptm", "interface_pae", "interface_plddt", "mean_plddt"):
        assert key in score.extras
    assert 0.0 <= float(score.extras["iptm"]) <= 1.0
    assert float(score.value) == float(score.extras["iptm"])


def test_missing_prediction_is_nan_not_crash():
    scorer = ProtenixConfidenceScorer(predictions_root=FIXTURE_ROOT)
    ex = BenchmarkExample(id="does__not_exist", binder_chains=("Q",),
                          target_chains=("E",), label="1", binder_format="vhh")
    assert scorer.score(ex).extras.get("missing") == "prediction"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_protenix_confidence_scorer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the extractor (sequence-resolved chains)**

Fill the `_KEY_*`/`_GLOB` constants from the Task 1 schema doc (single documented reconciliation point). Chains come from `chain_roles.json` if present else `resolve_chain_roles_by_sequence` — **never positional**.

```python
# src/mirage/scorers/protenix_confidence.py  (abridged — see Task 2 helpers)
from mirage.scorers._structure import (
    chain_residues, load_structure, read_chain_roles_json, resolve_chain_roles_by_sequence)
...
    def _score_real(self, example, out_dir, conf_path, struct_path):
        conf = json.loads(conf_path.read_text())
        iptm, ptm = float(conf[_KEY_IPTM]), float(conf[_KEY_PTM])
        pred = load_structure(struct_path)
        roles = read_chain_roles_json(out_dir / "chain_roles.json")
        binder_ids, target_ids = roles or resolve_chain_roles_by_sequence(
            pred, example.binder_chains, example.target_chains)
        binder_res = [r for c in binder_ids for r in chain_residues(pred, c)]
        target_res = [r for c in target_ids for r in chain_residues(pred, c)]
        inter_b, inter_t = _interface_masks(binder_res, target_res, INTERFACE_CUTOFF_A)
        pae = _load_pae(out_dir, conf)
        plddt = _per_residue_plddt(out_dir, conf, n_res=len(binder_res) + len(target_res))
        extras = {
            "iptm": iptm, "ptm": ptm,
            "interface_pae": _interface_pae(pae, len(binder_res), inter_b, inter_t),
            "interface_plddt": _interface_plddt(plddt, len(binder_res), inter_b, inter_t),
            "mean_plddt": float(np.nanmean(plddt)) if plddt.size else float("nan"),
        }
        return Score(example_id=example.id, scorer_name=self.name, value=iptm, extras=extras)
```
Add helpers `_interface_masks` (chunked heavy-atom 8 Å, as in `StructuralInterfaceScorer._distance_stats`), `_load_pae`, `_per_residue_plddt`, `_interface_pae` (mean of binder-iface × antigen-iface PAE block), `_interface_plddt` (mean pLDDT over the interface-residue union). Add `default_protenix_predictions_root()` to `_paths.py` → `<repo_root>/data/raw/predictions/protenix`. Wrap `score()` in try/except → `nan_score(... error=...)`; missing files → `nan_score(missing="prediction")`. Register + side-effect import in `scorers/__init__.py`.

- [ ] **Step 4: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_protenix_confidence_scorer.py -v && uv run ruff check && uv run mypy src/mirage
git add src/mirage/scorers/protenix_confidence.py src/mirage/scorers/__init__.py \
        src/mirage/_paths.py tests/test_protenix_confidence_scorer.py
git commit -m "Add Protenix confidence-internals scorer (sequence-resolved interface)"
```

---

## Task 4: `ProtenixPosePredictor` + SLURM runner (precomputed-MSA, chain_roles sidecar)

**Files:** Create `src/mirage/pose_predictors/protenix.py`, `scripts/slurm/run_protenix_chunk.py`, `scripts/slurm/predict_protenix.slurm`; modify `src/mirage/pose_predictors/__init__.py`. Test `tests/test_protenix_pose_predictor.py`.

- [ ] **Step 1: Write the failing test (staging + sbatch, no GPU)**

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
    assert manifest.n_rows == 2 and manifest.path.is_file()
    assert manifest.path.read_text().splitlines()[0].split("\t") == \
        ["example_id", "input_path", "out_dir"]


def test_stage_skips_cached(tmp_path):
    pred = ProtenixPosePredictor(output_root=tmp_path / "out", staged_root=tmp_path / "stage")
    (tmp_path / "out" / "p0").mkdir(parents=True)
    (tmp_path / "out" / "p0" / "rank1.pdb").write_text("X")
    manifest = pred.stage([_ex(0), _ex(1)])
    assert manifest.n_rows == 1 and manifest.n_already_cached == 1


def test_sbatch_command_array_and_account(tmp_path):
    pred = ProtenixPosePredictor(output_root=tmp_path / "out", staged_root=tmp_path / "stage")
    manifest = pred.stage([_ex(0), _ex(1), _ex(2)])
    cmd = pred.sbatch_command(manifest, chunk_size=2, max_concurrent=4)
    assert "--array=0-1%4" in cmd
    assert any("dbgoodma-goodman-laboratory" in c for c in cmd)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_protenix_pose_predictor.py -v` → FAIL.

- [ ] **Step 3: Implement the predictor (mirror `af2m.py`)**

`ProtenixPosePredictor` mirrors `AF2MPosePredictor`, deltas: manifest columns `example_id, input_path, out_dir`; `stage()` writes a per-example Protenix **JSON** input (binder chain first, antigen second; **templates disabled**; each chain references its precomputed MSA path by sequence hash — see Task 6) instead of a FASTA; `results_for` → `<out>/<id>/rank1.pdb`; defaults account `dbgoodma-goodman-laboratory`, partition `b200-mig45`. Reuse `_parse_sbatch_job_id`, `_clean_sequence`, chunk-array math. Add `protenix_from_env()`. Provide a module-level `sequence_hash(seq: str) -> str` (`hashlib.sha1(seq.encode()).hexdigest()[:16]`) used by both staging and MSA pre-compute so the JSON's MSA path and the cache key agree.

- [ ] **Step 4: SLURM runner + wrapper (precomputed-MSA, writes `chain_roles.json`)**

`scripts/slurm/run_protenix_chunk.py` (stdlib-only, mirror `run_af2m_chunk.py`): read manifest, slice chunk; for each row lacking `rank1.pdb`, shell out to Protenix in **precomputed-MSA mode** (the input JSON already points at cached `.a3m`; pass the schema-doc flag to disable the MSA server, e.g. `--use_msa_server false`), `--seeds 0`, templates off, into `out_dir`. Post-process: (1) symlink top-ranked CIF/PDB → `rank1.pdb`; (2) copy/normalize the confidence JSON to a stable name; (3) **write `chain_roles.json`** — load the output, call `resolve_chain_roles_by_sequence(struct, binder_seqs, target_seqs)` (carry the two submitted sequences as extra manifest columns so the runner has them), dump `{"binder": [...], "target": [...]}`. Same logging/skip-cached/return-code semantics as af2m. `predict_protenix.slurm`: `#SBATCH` account/partition/qos from spec, `conda activate protenix`, `python run_protenix_chunk.py "$1" "$SLURM_ARRAY_TASK_ID" "$2"`.

- [ ] **Step 5: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_protenix_pose_predictor.py -v && uv run ruff check && uv run mypy src/mirage
git add src/mirage/pose_predictors/protenix.py src/mirage/pose_predictors/__init__.py \
        scripts/slurm/run_protenix_chunk.py scripts/slurm/predict_protenix.slurm \
        tests/test_protenix_pose_predictor.py
git commit -m "Add ProtenixPosePredictor + SLURM runner (precomputed-MSA, chain_roles sidecar)"
```

---

## Task 5: `stage_protenix_pairs.py` — per-pair inputs + dedup unique-sequence manifest

**Files:** Create `scripts/stage_protenix_pairs.py`, `scripts/stage_champloo_protenix_pairs.py`. Test `tests/test_stage_protenix_pairs.py`.

- [ ] **Step 1: Write the failing test (examples + unique-seq dedup)**

```python
# tests/test_stage_protenix_pairs.py
import csv
from pathlib import Path

from scripts.stage_protenix_pairs import examples_from_pairs_csv, unique_sequences


def _write_pairs(p: Path):
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "binder_seq", "antigen_seq",
                                           "label", "antigen_cluster", "fold"])
        w.writeheader()
        # same binder QVQ reused across two pairs; antigen EVAL shared too
        w.writerow({"pair_id": "A__B", "binder_seq": "QVQ", "antigen_seq": "EVAL",
                    "label": "1", "antigen_cluster": "3", "fold": "0"})
        w.writerow({"pair_id": "A__B__neg0", "binder_seq": "QVQ", "antigen_seq": "WXYZ",
                    "label": "0", "antigen_cluster": "5", "fold": "0"})


def test_examples_from_pairs_csv(tmp_path):
    csv_path = tmp_path / "pairs.csv"; _write_pairs(csv_path)
    ex = list(examples_from_pairs_csv(csv_path))
    assert [e.id for e in ex] == ["A__B", "A__B__neg0"]
    assert ex[0].binder_chains == ("QVQ",) and ex[0].target_chains == ("EVAL",)


def test_unique_sequences_dedup(tmp_path):
    csv_path = tmp_path / "pairs.csv"; _write_pairs(csv_path)
    uniq = unique_sequences(csv_path)          # set of distinct sequences across all chains
    assert uniq == {"QVQ", "EVAL", "WXYZ"}      # QVQ counted once despite two pairs
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement staging**

`examples_from_pairs_csv(path)` yields `BenchmarkExample(id=pair_id, binder_chains=(binder_seq,), target_chains=(antigen_seq,), label=..., binder_format="vhh", metadata={antigen_cluster, fold})`. `unique_sequences(path)` returns the set of all distinct binder+antigen sequences. The Typer `main` calls `ProtenixPosePredictor.stage(...)` to write per-pair JSON inputs **and** writes `<staged_root>/unique_seqs.txt` (one sequence + its `sequence_hash` per line) for the MSA pre-compute. Each per-pair JSON references `msa_cache/<hash>.a3m` for each chain. `--submit` dispatches the array. Confirm `BenchmarkExample` accepts `metadata=` (per `scorers/base.py`).

- [ ] **Step 4: Champloo pairs builder**

`scripts/stage_champloo_protenix_pairs.py`: read the Champloo supplementary CSV, take the 91 cognate-diagonal positives, normalize sequences, cluster antigens (`features/clustering.py`), assign folds, and build matched k=5 cross-cluster negatives by importing `build_pairs` from `scripts/stage_sabdab_pairs.py` (refactor `build_pairs` to accept a generic positives list if needed). Emit `data/staged/protenix/champloo_pairs.csv` with the same six columns. Test: 91 positives + 455 negatives.

- [ ] **Step 5: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_stage_protenix_pairs.py -v && uv run ruff check && uv run mypy src/mirage
git add scripts/stage_protenix_pairs.py scripts/stage_champloo_protenix_pairs.py \
        tests/test_stage_protenix_pairs.py
git commit -m "Add Protenix pair staging + unique-sequence dedup manifest (MSA reuse)"
```

---

## Task 6: MSA pre-compute — run the search ONCE per unique sequence

Fixes the MSA fan-out trap: ~3,234 pairs reference only ~800 unique sequences. Compute each sequence's MSA exactly once into a shared cache; per-pair jobs consume `.a3m`, never search.

**Files:** Create `scripts/precompute_protenix_msa.py`, `scripts/slurm/precompute_msa.slurm`. Test `tests/test_precompute_msa.py`.

- [ ] **Step 1: Write the failing test (cache keying + skip-existing)**

```python
# tests/test_precompute_msa.py
from pathlib import Path

from scripts.precompute_protenix_msa import msa_targets, cache_path_for


def test_cache_path_is_hash_keyed(tmp_path):
    p = cache_path_for(tmp_path, "QVQ")
    assert p.parent == tmp_path and p.suffix == ".a3m"


def test_msa_targets_skips_already_cached(tmp_path):
    manifest = tmp_path / "unique_seqs.txt"
    manifest.write_text("QVQ\t<hash1>\nEVAL\t<hash2>\n")
    # pre-create QVQ's a3m
    (cache_path_for(tmp_path, "QVQ")).write_text(">x\nQVQ\n")
    todo = list(msa_targets(manifest, cache_dir=tmp_path))
    assert [seq for seq, _ in todo] == ["EVAL"]   # QVQ skipped (cached)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the pre-compute**

`cache_path_for(cache_dir, seq)` → `cache_dir / f"{sequence_hash(seq)}.a3m"` (import `sequence_hash` from `pose_predictors.protenix` so keys match staging exactly). `msa_targets(manifest, cache_dir)` yields `(seq, hash)` for rows whose `.a3m` is absent. The Typer `main` runs the MSA search for each todo sequence using the mechanism confirmed in Task 1 (Protenix's own MSA tool, or a local `mmseqs2` colabfold-style search) writing `<hash>.a3m`. Idempotent: re-runs skip cached. `precompute_msa.slurm` runs it as a SLURM array over the unique manifest (CPU partition or GPU as the MSA tool requires).

- [ ] **Step 4: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_precompute_msa.py -v && uv run ruff check && uv run mypy src/mirage
git add scripts/precompute_protenix_msa.py scripts/slurm/precompute_msa.slurm \
        tests/test_precompute_msa.py
git commit -m "Add MSA pre-compute: one search per unique sequence, hash-keyed a3m cache"
```

---

## Task 7: CDR-engagement extractor (TDD) — `features/cdr_engagement.py`

**Files:** Create `src/mirage/features/cdr_engagement.py`. Test `tests/test_cdr_engagement.py`.

IMGT CDR ranges: CDR1 27–38, CDR2 56–65, CDR3 105–117.

- [ ] **Step 1: Write the failing test (row-preserving fallback is the contract)**

```python
# tests/test_cdr_engagement.py
import numpy as np

from mirage.features.cdr_engagement import cdr_engagement_features, CDR_FEATURE_NAMES


def test_fallback_is_row_preserving_on_anarci_failure():
    feats = cdr_engagement_features(binder_seq="AAAAAAAAAA",
                                    binder_interface_residue_indices=np.array([0, 1, 2]),
                                    n_binder_residues=10)
    assert feats["cdr_mapping_ok"] == 0.0
    assert feats["cdr_contact_fraction"] == 0.0
    assert set(CDR_FEATURE_NAMES).issubset(feats.keys())
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the extractor**

Public `cdr_engagement_features(binder_seq, binder_interface_residue_indices, n_binder_residues) -> dict[str, float]`. Constant `CDR_FEATURE_NAMES = ("cdr_contact_fraction","cdr1_contact_fraction","cdr2_contact_fraction","cdr3_contact_fraction","cdr_mapping_ok")`; `_defaults()` = all 0.0. `_imgt_cdr_mask(binder_seq, n_residues)`: resolve HMMER via `_resolve_hmmer_bin()` (None → return None); `run_anarci([("q", binder_seq)], scheme="imgt", hmmerpath=hmmer_bin)`; from the numbering (`result[1][0]`) and details (`result[2][0]`, `query_start`), build a **dict of four boolean arrays** of length `n_residues` (`any`, `cdr1`, `cdr2`, `cdr3`) flagging residue indices whose IMGT number is in `_CDR_RANGES`; any exception → None. Public fn: if mask is None or no interface indices → `_defaults()`; else fill `cdr_mapping_ok=1.0`, `cdr_contact_fraction=masks["any"][iface].mean()`, and per-CDR fractions. **Never raises.**

- [ ] **Step 4: Run tests (with HMMER) + lint + types; commit**

```bash
bash scripts/install_hmmer.sh   # if .tools/hmmer absent
uv run pytest tests/test_cdr_engagement.py -v && uv run ruff check && uv run mypy src/mirage
git add src/mirage/features/cdr_engagement.py tests/test_cdr_engagement.py
git commit -m "Add CDR-engagement interface features with row-preserving ANARCI fallback"
```

---

## Task 8: `extract_mc_features.py` — fault-tolerant feature assembler (sequence-resolved chains)

Fixes the array fault-tolerance trap: never crash the assembly; log missing/corrupt predictions and continue. Uses sequence-resolved chains end-to-end.

**Files:** Create `scripts/extract_mc_features.py`. Test `tests/test_extract_mc_features.py`.

Geometry subset (from `StructuralInterfaceScorer(chain_resolution="sequence")`): `n_interface_residues_binder`, `n_interface_residues_target`, `buried_sasa_proxy_a2`, `atom_contacts_5a`, `shape_complementarity_proxy`, `atom_clash_fraction_2a`. Confidence subset: `iptm`, `ptm`, `interface_pae`, `interface_plddt`, `mean_plddt`. CDR subset: the five `CDR_FEATURE_NAMES`. Passthrough: `pair_id`, `label`, `antigen_cluster`, `fold`. Plus `prediction_present`.

- [ ] **Step 1: Write the failing test (assembly + fault tolerance)**

```python
# tests/test_extract_mc_features.py
from pathlib import Path

from scripts.extract_mc_features import FEATURE_COLUMNS, assemble_row, build_rows


def test_feature_columns_complete():
    for col in ("pair_id", "label", "antigen_cluster", "fold", "iptm", "interface_pae",
                "n_interface_residues_binder", "cdr_contact_fraction", "cdr_mapping_ok",
                "prediction_present"):
        assert col in FEATURE_COLUMNS


def test_assemble_row_marks_missing_prediction(tmp_path):
    row = {"pair_id": "X__Y", "binder_seq": "QVQ", "antigen_seq": "EVAL",
           "label": "1", "antigen_cluster": "3", "fold": "0"}
    out = assemble_row(row, predictions_root=tmp_path)   # nothing on disk
    assert out["pair_id"] == "X__Y" and out["prediction_present"] == "0"
    assert out["iptm"] == ""                              # blank, never fabricated


def test_build_rows_logs_failures_and_continues(tmp_path, monkeypatch):
    # one good-looking row, one that raises inside assemble_row
    rows = [{"pair_id": "OK", "binder_seq": "Q", "antigen_seq": "E", "label": "1",
             "antigen_cluster": "1", "fold": "0"},
            {"pair_id": "BOOM", "binder_seq": "Q", "antigen_seq": "E", "label": "0",
             "antigen_cluster": "2", "fold": "0"}]
    import scripts.extract_mc_features as m

    def fake_assemble(row, predictions_root):
        if row["pair_id"] == "BOOM":
            raise ValueError("truncated json")
        return {**{c: "" for c in m.FEATURE_COLUMNS}, "pair_id": "OK",
                "prediction_present": "1"}
    monkeypatch.setattr(m, "assemble_row", fake_assemble)
    failed = tmp_path / "failed_pairs.txt"
    built = build_rows(rows, predictions_root=tmp_path, failed_log=failed)
    assert [r["pair_id"] for r in built] == ["OK"]        # CSV still produced for successes
    assert "BOOM" in failed.read_text()                   # failure logged, not raised
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the assembler**

`assemble_row(row, predictions_root)`: build the `BenchmarkExample`; if `<root>/<pair_id>/rank1.pdb` absent → return all-blank feature cols + `prediction_present="0"`. Else score with `StructuralInterfaceScorer(predictions_root, chain_resolution="sequence")` and `ProtenixConfidenceScorer(predictions_root)`; recompute the binder interface-residue indices from the **sequence-resolved** binder chain (via `_structure.interface_residue_indices` / the geometry scorer's interface mask) and feed them to `cdr_engagement_features`; merge the three feature dicts; `prediction_present="1"`. `build_rows(rows, predictions_root, failed_log)`: loop, wrap each `assemble_row` in `try/except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError)` → append `pair_id` to `failed_log`, `continue`; collect successes. The individual scorers already catch internally and return `nan_score`, so `build_rows` is the belt-and-suspenders layer guaranteeing the run finishes. The Typer `main` writes `data/staged/mc/<dataset>_features.csv` and prints `built=N failed=M` (M from the failed log).

- [ ] **Step 4: Run tests + lint + types; commit**

```bash
uv run pytest tests/test_extract_mc_features.py -v && uv run ruff check && uv run mypy src/mirage
git add scripts/extract_mc_features.py tests/test_extract_mc_features.py
git commit -m "Add fault-tolerant M-C feature assembler (sequence-resolved chains, failed_pairs log)"
```

---

## Task 9: Run the campaign + cognate≫shuffled validation (operational; B1→B2 boundary)

Operational (SLURM + GPU): commands + acceptance criteria, not unit tests. **Order: stage → MSA pre-compute → predict → validate → assemble.**

- [ ] **Step 1: Stage both pair sets (writes per-pair JSON + unique-seq manifests)**

```bash
uv run python scripts/stage_champloo_protenix_pairs.py \
  --champloo-table ../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv \
  --output data/staged/protenix/champloo_pairs.csv
for ds in sabdab champloo; do
  pairs=$([ $ds = sabdab ] && echo data/staged/sabdab/sabdab_pairs.csv || echo data/staged/protenix/champloo_pairs.csv)
  uv run python scripts/stage_protenix_pairs.py --pairs "$pairs" \
    --output-root data/raw/predictions/protenix/$ds \
    --staged-root data/staged/protenix/$ds   # no --submit yet: MSAs first
done
```
Expected: per-pair JSON written; `data/staged/protenix/{sabdab,champloo}/unique_seqs.txt` emitted.

- [ ] **Step 2: Pre-compute MSAs ONCE per unique sequence (the fan-out guard)**

```bash
cat data/staged/protenix/sabdab/unique_seqs.txt data/staged/protenix/champloo/unique_seqs.txt \
  | sort -u > data/staged/protenix/all_unique_seqs.txt
wc -l data/staged/protenix/all_unique_seqs.txt   # expect ~800, NOT ~3234
sbatch scripts/slurm/precompute_msa.slurm data/staged/protenix/all_unique_seqs.txt \
       data/staged/protenix/msa_cache
```
Acceptance: the unique count is ~800 (sanity that dedup works); MSA cache fills with `<hash>.a3m`. **Do not submit predictions until the cache is populated.**

- [ ] **Step 3: Submit both prediction campaigns (precomputed-MSA mode)**

```bash
for ds in sabdab champloo; do
  pairs=$([ $ds = sabdab ] && echo data/staged/sabdab/sabdab_pairs.csv || echo data/staged/protenix/champloo_pairs.csv)
  uv run python scripts/stage_protenix_pairs.py --pairs "$pairs" \
    --output-root data/raw/predictions/protenix/$ds \
    --staged-root data/staged/protenix/$ds --submit --chunk-size 8 --max-concurrent 16
done
```
Expected: `staged rows=2688` (sabdab) / `546` (champloo) then job ids. Monitor `squeue -u $USER`. (Re-running stage is idempotent: cached predictions are skipped.)

- [ ] **Step 4: Wait + check coverage**

```bash
uv run python - <<'PY'
from pathlib import Path
for name, n in [("sabdab", 2688), ("champloo", 546)]:
    root = Path(f"data/raw/predictions/protenix/{name}")
    done = sum(1 for d in root.iterdir() if (d / "rank1.pdb").is_file()) if root.exists() else 0
    print(name, done, "/", n)
PY
```
Acceptance: ≥99% coverage. Re-submit for stragglers (idempotent).

- [ ] **Step 5: §4 install GATE — cognate ≫ shuffled ipTM on the Champloo diagonal**

```bash
uv run python - <<'PY'
import csv, statistics
from pathlib import Path
from mirage.scorers.protenix_confidence import ProtenixConfidenceScorer
from mirage.scorers.base import BenchmarkExample
sc = ProtenixConfidenceScorer(predictions_root=Path("data/raw/predictions/protenix/champloo"))
pos, neg = [], []
for r in csv.DictReader(open("data/staged/protenix/champloo_pairs.csv")):
    ex = BenchmarkExample(id=r["pair_id"], binder_chains=(r["binder_seq"],),
                          target_chains=(r["antigen_seq"],), label=r["label"], binder_format="vhh")
    v = sc.score(ex).extras.get("iptm")
    if isinstance(v, float): (pos if r["label"] == "1" else neg).append(v)
print("cognate median", statistics.median(pos), "shuffled median", statistics.median(neg))
PY
```
Acceptance: cognate median clearly above shuffled. If they overlap → STOP, the predictor/inputs are wrong; do not proceed to B2.

- [ ] **Step 6: Materialize cached feature CSVs (fault-tolerant)**

```bash
uv run python scripts/extract_mc_features.py --pairs data/staged/sabdab/sabdab_pairs.csv \
  --predictions-root data/raw/predictions/protenix/sabdab \
  --output data/staged/mc/sabdab_features.csv --failed-log data/staged/mc/sabdab_failed.txt
uv run python scripts/extract_mc_features.py --pairs data/staged/protenix/champloo_pairs.csv \
  --predictions-root data/raw/predictions/protenix/champloo \
  --output data/staged/mc/champloo_features.csv --failed-log data/staged/mc/champloo_failed.txt
```
Acceptance: both CSVs have one row per **successful** pair with the full `FEATURE_COLUMNS` header; `*_failed.txt` lists any stragglers to requeue; record the CDR-mapping failure rate (high on negatives is a finding per the spec).

- [ ] **Step 7: Document + commit the run record (not the large blobs)**

Update `docs/datasets/dataset-registry.md` and write `results/published/mc_campaign_record.md` (job ids, unique-seq count, coverage, cognate/shuffled ipTM medians, CDR-mapping failure rate, requeued pairs). `data/raw/predictions/` + `data/staged/` are gitignored — commit only docs.
```bash
git add docs/datasets/dataset-registry.md results/published/mc_campaign_record.md
git commit -m "Record M-C Protenix campaign: MSA reuse, coverage, ipTM gate, CDR failure rate"
```

---

## Phase B2 preview (separate plan, authored after Task 9)

Once `data/staged/mc/{sabdab,champloo}_features.csv` exist, Phase B2 adds: `paired_delta_bootstrap` in `eval/gate.py` (ΔAUROC/Δprecision on shared resampled rows); a rung-ladder trainer calling `train_ms` over the four column subsets (rung 0 `[iptm]` → +confidence → +geometry → +CDR); `analyze_mc_indist.py` (SAbDab OOF, floor-check vs 0.496, paired rung-deltas, `--random-folds` contrast, SHAP); `analyze_mc_cross_regime.py` (bidirectional 2a SAbDab→Champloo / 2b Champloo→SAbDab, antigen dedup guard, `FrozenGate`); `analyze_mc_af3_companion.py` (Champloo AF3 rung-0→2 + AF3-vs-Protenix ipTM); `results/published/mc_structure_summary.md`. Its exact columns come from Task 8's `FEATURE_COLUMNS`, which is why it is deferred.

---

## Self-Review

**Spec coverage:** Protenix predictor + templates-off (T4) ✓; single-predictor campaign over both sets, full 2,688 SAbDab + 546 Champloo (T5,T9) ✓; predict-the-shuffled-pair negatives (T5,T9) ✓; **MSA caching per unique sequence — now an explicit pre-compute (T6) + dedup manifest (T5), with a ~800-not-3234 sanity gate (T9 Step 2)** ✓; rung-0→3 feature substrate — confidence internals (T3), geometry subset (T8 via refactored scorer), CDR engagement with row-preserving fallback (T7) ✓; install-validation cognate≫shuffled gate (T1+T9) ✓; torch-free invariant ✓. Modeling/eval = Phase B2.

**Three hardening fixes, each tied to a task:** MSA fan-out → T5 dedup manifest + T6 one-search-per-sequence + T9 precompute-before-predict ordering. Chain reordering → T2 `resolve_chain_roles_by_sequence` + geometry-scorer chain injection + T4 `chain_roles.json` sidecar; all Protenix feature paths (T3, T8) resolve by sequence, never positionally. Fault tolerance → T8 `build_rows` try/except → `failed_pairs.txt`, plus the scorers' internal `nan_score` guards.

**Placeholder scan:** the only empirical-reconciliation points are the Protenix `_KEY_*`/`_GLOB` constants and the MSA/templates CLI flags — all resolved by the Task 1 schema doc + committed fixture, not silent TODOs.

**Type consistency:** `resolve_chain_roles_by_sequence(structure, binder_seqs, target_seqs) -> tuple[list[str], list[str]]` is defined in T2 and consumed in T3, T4, T8; `sequence_hash` defined once in T4 and reused in T5/T6 so MSA cache keys match staging; `chain_roles.json` schema (`{"binder":[...],"target":[...]}`) written in T4, read by `read_chain_roles_json` in T2 and used in T3/T8; `CDR_FEATURE_NAMES` defined in T7, consumed in T8; `StructuralInterfaceScorer(chain_resolution="sequence")` introduced in T2, used in T8.

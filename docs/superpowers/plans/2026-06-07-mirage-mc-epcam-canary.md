# mirage M-C EpCAM real-negative canary — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score 14 real designed EpCAM VHHs (8 functional / 6 non-functional, CAR-T killing labels) plus ~70 shuffled negatives through Protenix, then evaluate the **frozen SAbDab M-C gate** on them — testing whether ipTM generalizes to the designed-binder deployment regime and whether interface geometry/CDR add anything there.

**Architecture:** Two new pure-Python scripts mirror existing M-C scripts. `stage_epcam_protenix_pairs.py` builds an EpCAM pairs CSV (14 positives + k=5 shuffled negatives per VHH, antigens drawn from the SAbDab pool with an antigen-cluster dedup guard) in the schema the existing Protenix staging/feature pipeline already consumes. `analyze_mc_epcam.py` re-fits the SAbDab rung-0 and rung-3 gates on the committed SAbDab features and applies them **unchanged** (frozen transfer, never trained on EpCAM) to the EpCAM features, producing three reads: primary designed-binders-vs-shuffled AUROC + paired Δ(R3−R0), the frozen-threshold calibration check, and the exploratory functional-vs-non-functional killing read. The GPU prediction campaign reuses the B1 Protenix pipeline verbatim.

**Tech Stack:** Python 3.11, numpy, typer; uv-managed mirage env (torch-free); Protenix 2.0.0 in the separate `protenix` conda env on PARCC B200 (shell-out / SLURM); pytest + ruff + mypy.

---

## Prerequisites (must exist locally before starting)

These are gitignored outputs from the M-S / M-C B1 work, present on PARCC:

- `data/staged/sabdab/sabdab_pairs.csv` — SAbDab pairs (the antigen pool source).
- `data/staged/mc/sabdab_features.csv` — SAbDab M-C feature CSV (the frozen-gate training data).
- `results/published/mc_sabdab_model.json` — the persisted rung-3 SAbDab gate (sanity reference).
- `../abdisc-data/epcam/epcam_killing_labels.csv` — owner-authored `vhh_id,vhh_sequence,label` (Good/Bad) for the 14 VHHs. The `epcam_killing` loader already reads this.
- Protenix env on PARCC per `results/published/mc_campaign_record.md` + the M-C B1 `HANDOFF.md` (env vars, cu128 torch, weight cache at `PROTENIX_ROOT_DIR`).

If `epcam_killing_labels.csv` is absent, stop and surface it — it is owner-authored, not derivable.

---

## File Structure

- **Create** `scripts/stage_epcam_protenix_pairs.py` — EpCAM pairs CSV builder (`epcam_antigen_negative_pool`, `build_epcam_pairs`, `load_epcam_positives`, Typer CLI). One responsibility: turn the 14 labeled VHHs + the SAbDab antigen pool into a leakage-guarded pairs CSV in the standard 6-column schema.
- **Create** `tests/test_stage_epcam_protenix_pairs.py` — unit tests for the three helpers.
- **Create** `scripts/analyze_mc_epcam.py` — frozen-transfer analysis (`fit_sabdab_rung`, `vhh_id_from_pair`, `killing_label_map`, `analyze_epcam`, Typer/argparse CLI → JSON). One responsibility: apply the frozen SAbDab gate to EpCAM features and emit the three reads.
- **Create** `tests/test_analyze_mc_epcam.py` — unit + smoke-integration tests on tiny synthetic CSVs.
- **Create** (by the run) `data/staged/epcam/epcam_protenix_pairs.csv`, `data/staged/mc/epcam_features.csv` (gitignored), `results/published/mc_epcam_canary.json`, `results/published/mc_epcam_canary_summary.md` (committed).

No existing files are modified (scripts are standalone; the mirage package is reused read-only).

---

## Task 1: EpCAM pair staging — negative pool + dedup guard

**Files:**
- Create: `scripts/stage_epcam_protenix_pairs.py`
- Test: `tests/test_stage_epcam_protenix_pairs.py`

- [ ] **Step 1: Write the failing test for the negative pool dedup guard**

```python
# tests/test_stage_epcam_protenix_pairs.py
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stage_epcam_protenix_pairs.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage_epcam_protenix_pairs", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage_epcam_protenix_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_negative_pool_excludes_epcam_cluster_antigens():
    mod = _load()
    epcam = "MARTGYHIKLPQRSTWVYACDEFGHIKLMNPQ"  # stand-in EpCAM ECD
    near_epcam = epcam[:-1] + "A"  # ~identical -> same cluster, must be excluded
    far1 = "WWWWWWWWWWQQQQQQQQQQYYYYYYYYYYEE"
    far2 = "CCCCCCCCCCDDDDDDDDDDFFFFFFFFFFGG"
    pool = mod.epcam_antigen_negative_pool([near_epcam, far1, far2], epcam, max_identity=0.9)
    assert near_epcam not in pool
    assert far1 in pool and far2 in pool
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage && uv run pytest tests/test_stage_epcam_protenix_pairs.py::test_negative_pool_excludes_epcam_cluster_antigens -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (script/function not defined).

- [ ] **Step 3: Create the script with the negative-pool helper**

```python
# scripts/stage_epcam_protenix_pairs.py
"""Build the EpCAM real-negative canary pairs CSV for the M-C structure track.

14 designed EpCAM VHHs (CAR-T killing labels) x EpCAM ECD are the positives;
negatives are predict-the-shuffled-pair: each VHH x k wrong antigens drawn from
the SAbDab antigen pool, with the EpCAM antigen cluster excluded (leakage guard).
Emits the standard 6-column pairs schema consumed by stage_protenix_pairs.py and
extract_mc_features.py.

Use::

    uv run python scripts/stage_epcam_protenix_pairs.py \\
        --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \\
        --output data/staged/epcam/epcam_protenix_pairs.csv \\
        --k 5
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from mirage.benchmark.epcam_killing import EpCAMKillingLoader
from mirage.features.clustering import cluster_antigens
from mirage.features.normalize import normalize_antigen, normalize_binder

_FIELDNAMES = ["pair_id", "binder_seq", "antigen_seq", "label", "antigen_cluster", "fold"]


def epcam_antigen_negative_pool(
    sabdab_antigens: list[str], epcam_antigen: str, *, max_identity: float = 0.9
) -> list[str]:
    """SAbDab antigens that do NOT share a cluster with the EpCAM antigen.

    Clusters the pooled (EpCAM-first) antigen sequences and drops any SAbDab
    antigen whose cluster equals the EpCAM cluster. Multi-chain antigens (``:``)
    are clustered on their concatenated sequence but returned verbatim.
    """
    pooled = [epcam_antigen, *sabdab_antigens]
    clusters = cluster_antigens([a.replace(":", "") for a in pooled], max_identity=max_identity)
    epcam_cluster = clusters[0]
    return [
        ag
        for ag, c in zip(sabdab_antigens, clusters[1:], strict=True)
        if c != epcam_cluster
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_stage_epcam_protenix_pairs.py::test_negative_pool_excludes_epcam_cluster_antigens -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_epcam_protenix_pairs.py tests/test_stage_epcam_protenix_pairs.py
git commit -m "EpCAM canary staging: antigen-cluster dedup guard for the negative pool"
```

---

## Task 2: EpCAM pair staging — build_epcam_pairs

**Files:**
- Modify: `scripts/stage_epcam_protenix_pairs.py`
- Test: `tests/test_stage_epcam_protenix_pairs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_epcam_pairs_shape_and_ids():
    mod = _load()
    positives = [
        ("10", "QVQLVESGGG", "EPCAMSEQ", "functional"),
        ("14", "EVQLVESGGA", "EPCAMSEQ", "nonfunctional"),
    ]
    pool = [f"WRONGANTIGEN{i}" for i in range(20)]
    rows = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    # 2 positives + 2*5 negatives
    assert len(rows) == 12
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    assert len(pos) == 2 and len(neg) == 10
    # positive pair_id convention + antigen is EpCAM
    assert {r["pair_id"] for r in pos} == {"epcam-10__epcam", "epcam-14__epcam"}
    assert all(r["antigen_seq"] == "EPCAMSEQ" for r in pos)
    # negative pair_id convention + antigen drawn from the pool (never EpCAM)
    assert all(r["pair_id"].endswith(tuple(f"__neg{j}" for j in range(5))) for r in neg)
    assert all(r["antigen_seq"] in pool for r in neg)
    # schema columns present and exact
    assert all(set(r) == set(mod._FIELDNAMES) for r in rows)


def test_build_epcam_pairs_is_deterministic():
    mod = _load()
    positives = [("10", "QVQLVESGGG", "EPCAMSEQ", "functional")]
    pool = [f"A{i}" for i in range(50)]
    a = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    b = mod.build_epcam_pairs(positives, pool, k=5, seed=7)
    assert [r["antigen_seq"] for r in a] == [r["antigen_seq"] for r in b]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_stage_epcam_protenix_pairs.py -k build_epcam_pairs -v`
Expected: FAIL — `AttributeError: build_epcam_pairs`.

- [ ] **Step 3: Add `build_epcam_pairs` to the script**

Insert after `epcam_antigen_negative_pool`:

```python
def build_epcam_pairs(
    positives: list[tuple[str, str, str, str]],
    negative_pool: list[str],
    *,
    k: int,
    seed: int,
) -> list[dict[str, str]]:
    """Positive rows (VHH x EpCAM) + k shuffled negatives (VHH x pool antigen) each.

    ``positives`` items are ``(vhh_id, binder_seq, epcam_antigen_seq, killing_label)``;
    ``killing_label`` is not written to the CSV (re-derived at analysis time from the
    killing labels) but is accepted so callers pass the full positive record. Negative
    antigens are sampled without replacement per VHH. ``antigen_cluster`` and ``fold``
    are constant ``"0"`` — EpCAM is a frozen-transfer TEST set only, never trained or
    OOF-split, so neither column is consumed downstream.
    """
    if k > len(negative_pool):
        raise ValueError(f"k={k} exceeds negative pool size {len(negative_pool)}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for vhh_id, binder, antigen, _killing in positives:
        rows.append(
            {
                "pair_id": f"epcam-{vhh_id}__epcam",
                "binder_seq": binder,
                "antigen_seq": antigen,
                "label": "1",
                "antigen_cluster": "0",
                "fold": "0",
            }
        )
        idx = rng.choice(len(negative_pool), size=k, replace=False)
        for j, ai in enumerate(idx):
            rows.append(
                {
                    "pair_id": f"epcam-{vhh_id}__neg{j}",
                    "binder_seq": binder,
                    "antigen_seq": negative_pool[int(ai)],
                    "label": "0",
                    "antigen_cluster": "0",
                    "fold": "0",
                }
            )
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_stage_epcam_protenix_pairs.py -k build_epcam_pairs -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_epcam_protenix_pairs.py tests/test_stage_epcam_protenix_pairs.py
git commit -m "EpCAM canary staging: build_epcam_pairs (14 positives + k shuffled negatives)"
```

---

## Task 3: EpCAM pair staging — positives loader + CLI + leakage assertion

**Files:**
- Modify: `scripts/stage_epcam_protenix_pairs.py`
- Test: `tests/test_stage_epcam_protenix_pairs.py`

- [ ] **Step 1: Write the failing test for `load_epcam_positives`**

```python
def test_load_epcam_positives_maps_label_and_normalizes(tmp_path):
    mod = _load()
    csv_path = tmp_path / "killing.csv"
    csv_path.write_text(
        "vhh_id,vhh_sequence,label\n"
        "10,QVQLVESGGGLVQPGGSLRLSCAAS,Good\n"
        "14,EVQLVESGGGLVQPGGSLRLSCAAS,Bad\n"
    )
    pos = mod.load_epcam_positives(csv_path)
    assert [p[0] for p in pos] == ["10", "14"]
    assert [p[3] for p in pos] == ["functional", "nonfunctional"]
    # antigen is the EpCAM ECD constant (normalized), identical across positives
    assert pos[0][2] == pos[1][2] and len(pos[0][2]) > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_stage_epcam_protenix_pairs.py -k load_epcam_positives -v`
Expected: FAIL — `AttributeError: load_epcam_positives`.

- [ ] **Step 3: Add `load_epcam_positives` + the Typer CLI**

Append to the script:

```python
def load_epcam_positives(killing_labels: Path) -> list[tuple[str, str, str, str]]:
    """Load the 14 labeled VHHs as ``(vhh_id, binder, epcam_antigen, killing_label)``.

    Sequences are normalized at the featurization boundary (binder -> ANARCI IMGT
    variable domain; antigen -> signal-peptide/His-tag strip), exactly as the M-C
    feature pipeline does. ``Good`` -> ``functional``, ``Bad`` -> ``nonfunctional``.
    """
    loader = EpCAMKillingLoader(killing_labels)
    out: list[tuple[str, str, str, str]] = []
    for ex in loader.load():
        binder = normalize_binder(ex.binder_chains[0])
        antigen = normalize_antigen(ex.target_chains[0])
        killing = "functional" if ex.label == "BIND" else "nonfunctional"
        out.append((str(ex.metadata["vhh_id"]), binder, antigen, killing))
    return out


def sabdab_antigen_pool(sabdab_pairs: Path) -> list[str]:
    """Unique cognate antigen sequences from the SAbDab pairs CSV (label==1 rows)."""
    with sabdab_pairs.open(newline="") as fh:
        antigens = [r["antigen_seq"] for r in csv.DictReader(fh) if r["label"] == "1"]
    return list(dict.fromkeys(antigens))


app = typer.Typer(add_completion=False, help="Build the EpCAM canary pairs CSV.")


@app.command()
def main(
    killing_labels: Annotated[Path, typer.Option("--killing-labels", help="epcam_killing_labels.csv")],
    sabdab_pairs: Annotated[Path, typer.Option("--sabdab-pairs", help="SAbDab pairs CSV (antigen pool)")],
    output: Annotated[Path, typer.Option("--output", help="Output pairs CSV")],
    k: Annotated[int, typer.Option(help="Negatives per positive")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 20260607,
    max_identity: Annotated[float, typer.Option("--max-identity", help="Dedup identity")] = 0.9,
) -> None:
    """Stage EpCAM positives + dedup'd shuffled negatives; write to --output."""
    positives = load_epcam_positives(killing_labels)
    epcam_antigen = positives[0][2]

    # Leakage guard: none of the 14 designed VHHs may appear in SAbDab training.
    with sabdab_pairs.open(newline="") as fh:
        sabdab_binders = {r["binder_seq"] for r in csv.DictReader(fh)}
    overlap = {p[0] for p in positives if p[1] in sabdab_binders}
    if overlap:
        raise SystemExit(f"Leakage: EpCAM VHH(s) {sorted(overlap)} present in SAbDab binders")

    pool = epcam_antigen_negative_pool(
        sabdab_antigen_pool(sabdab_pairs), epcam_antigen, max_identity=max_identity
    )
    if len(pool) < k:
        raise SystemExit(f"Negative pool too small ({len(pool)}) for k={k}")

    rows = build_epcam_pairs(positives, pool, k=k, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    typer.echo(
        f"positives={n_pos} negatives={len(rows) - n_pos} total={len(rows)} "
        f"pool={len(pool)} (dedup max_identity={max_identity})"
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run the full staging test file + lint/type**

Run: `uv run pytest tests/test_stage_epcam_protenix_pairs.py -v && uv run ruff check scripts/stage_epcam_protenix_pairs.py && uv run mypy src/mirage`
Expected: all tests PASS; ruff clean; mypy clean (mypy targets `src/`; the script is checked via its imports).

- [ ] **Step 5: Commit**

```bash
git add scripts/stage_epcam_protenix_pairs.py tests/test_stage_epcam_protenix_pairs.py
git commit -m "EpCAM canary staging: positives loader, SAbDab antigen pool, CLI + leakage guard"
```

---

## Task 4: EpCAM analysis — frozen-transfer reads (pure functions)

**Files:**
- Create: `scripts/analyze_mc_epcam.py`
- Test: `tests/test_analyze_mc_epcam.py`

- [ ] **Step 1: Write the failing tests for the small helpers**

```python
# tests/test_analyze_mc_epcam.py
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_mc_epcam.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_mc_epcam", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_mc_epcam"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vhh_id_from_pair():
    mod = _load()
    assert mod.vhh_id_from_pair("epcam-10__epcam") == "10"
    assert mod.vhh_id_from_pair("epcam-74__epcam") == "74"


def test_killing_label_map(tmp_path):
    mod = _load()
    p = tmp_path / "k.csv"
    p.write_text("vhh_id,vhh_sequence,label\n10,AAAA,Good\n14,CCCC,Bad\n")
    m = mod.killing_label_map(p)
    assert m == {"10": 1, "14": 0}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_analyze_mc_epcam.py -k "vhh_id_from_pair or killing_label_map" -v`
Expected: FAIL — script/functions not defined.

- [ ] **Step 3: Create the script with the helpers + the core `analyze_epcam`**

```python
# scripts/analyze_mc_epcam.py
"""M-C EpCAM real-negative canary: apply the frozen SAbDab gate to EpCAM, unchanged.

Re-fits the SAbDab rung-0 and rung-3 gates on the committed SAbDab feature CSV
(full-fit; rung-3 reproduces mc_sabdab_model.json), then applies them — never
trained on EpCAM — to the EpCAM feature CSV. Three reads:

  1. PRIMARY (threshold-free): AUROC of rung-0 (ipTM) and rung-3 on
     designed-binders-vs-shuffled, with the paired Delta(R3-R0) bootstrap CI.
  2. CALIBRATION: the frozen rung-3 (and rung-0) gate at its SAbDab P=0.90
     threshold applied unchanged -> precision/recall/specificity + CIs.
  3. SECONDARY (exploratory): among the 14 positives, the rung-3 mirage score for
     the 8 functional vs 6 non-functional. N=14, killing != binding -> descriptive
     only, NO inferential claim.

Use::

    uv run python scripts/analyze_mc_epcam.py \\
        --sabdab-features data/staged/mc/sabdab_features.csv \\
        --epcam-features data/staged/mc/epcam_features.csv \\
        --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \\
        --output results/published/mc_epcam_canary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mirage.eval.gate import auroc, paired_delta_bootstrap
from mirage.eval.orthogonal import evaluate_frozen_gate
from mirage.features.mc_rungs import (
    fit_rung_model,
    folds_array,
    labels_array,
    read_feature_csv,
    rung_matrix,
)
from mirage.model.ms import MsModel

_TRANSFER_RUNG = 3
_CONTRAST_RUNG = 0


def vhh_id_from_pair(pair_id: str) -> str:
    """'epcam-10__epcam' -> '10'."""
    return pair_id.removeprefix("epcam-").split("__")[0]


def killing_label_map(killing_labels: Path) -> dict[str, int]:
    """vhh_id -> 1 (Good/functional) / 0 (Bad/non-functional)."""
    with killing_labels.open(newline="") as fh:
        return {r["vhh_id"]: (1 if r["label"].strip() == "Good" else 0) for r in csv.DictReader(fh)}


def fit_sabdab_rung(
    sab_rows: list[dict[str, str]], *, rung: int, l2: float, target_precision: float
) -> MsModel:
    """Full-fit a frozen SAbDab rung gate (rung-3 reproduces mc_sabdab_model.json)."""
    x, names = rung_matrix(sab_rows, rung=rung)
    y = labels_array(sab_rows)
    folds = folds_array(sab_rows)
    model, _ = fit_rung_model(
        x, y, folds, feature_names=names, l2=l2, target_precision=target_precision
    )
    return model


def analyze_epcam(
    sabdab_features: Path,
    epcam_features: Path,
    killing_labels: Path,
    *,
    l2: float,
    target_precision: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    sab_rows = read_feature_csv(sabdab_features)
    epc_rows = read_feature_csv(epcam_features)
    y = labels_array(epc_rows)

    m0 = fit_sabdab_rung(sab_rows, rung=_CONTRAST_RUNG, l2=l2, target_precision=target_precision)
    m3 = fit_sabdab_rung(sab_rows, rung=_TRANSFER_RUNG, l2=l2, target_precision=target_precision)

    x0, _ = rung_matrix(epc_rows, rung=_CONTRAST_RUNG)
    x3, _ = rung_matrix(epc_rows, rung=_TRANSFER_RUNG)
    s0 = m0.predict_logit(x0)
    s3 = m3.predict_logit(x3)

    # Read 1 — primary, threshold-free
    delta_point, delta_lo, delta_hi = paired_delta_bootstrap(
        s3, s0, y, statistic=auroc, n_boot=n_boot, seed=seed
    )
    primary = {
        "rung0_auroc": auroc(s0, y),
        "rung3_auroc": auroc(s3, y),
        "delta_auroc_r3_minus_r0": {"point": delta_point, "ci": [delta_lo, delta_hi]},
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }

    # Read 2 — calibration (frozen SAbDab P=target_precision threshold, unchanged)
    calibration = {
        "rung3": evaluate_frozen_gate(m3, x3, y, n_boot=n_boot, seed=seed),
        "rung0_contrast": evaluate_frozen_gate(m0, x0, y, n_boot=n_boot, seed=seed),
        "frozen_threshold_rung3": m3.threshold,
        "frozen_threshold_rung0": m0.threshold,
    }

    # Read 3 — secondary, exploratory (functional vs non-functional, N=14)
    kmap = killing_label_map(killing_labels)
    pos_idx = [i for i, r in enumerate(epc_rows) if r["label"] == "1"]
    func = np.array([kmap[vhh_id_from_pair(epc_rows[i]["pair_id"])] for i in pos_idx], dtype=int)
    pos_scores = s3[np.array(pos_idx, dtype=int)]
    secondary = {
        "caveat": "N=14, killing != binding, descriptive only — no inferential claim",
        "n_functional": int((func == 1).sum()),
        "n_nonfunctional": int((func == 0).sum()),
        "killing_auroc_rung3": auroc(pos_scores, func),
        "functional_scores": [float(v) for v in pos_scores[func == 1]],
        "nonfunctional_scores": [float(v) for v in pos_scores[func == 0]],
    }

    return {"primary": primary, "calibration": calibration, "secondary_killing": secondary}
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `uv run pytest tests/test_analyze_mc_epcam.py -k "vhh_id_from_pair or killing_label_map" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_mc_epcam.py tests/test_analyze_mc_epcam.py
git commit -m "EpCAM canary analysis: helpers + analyze_epcam (frozen transfer, three reads)"
```

---

## Task 5: EpCAM analysis — smoke-integration test + CLI

**Files:**
- Modify: `scripts/analyze_mc_epcam.py`
- Test: `tests/test_analyze_mc_epcam.py`

- [ ] **Step 1: Write the failing smoke-integration test**

This builds tiny synthetic SAbDab + EpCAM feature CSVs (with the exact `FEATURE_COLUMNS` schema) where ipTM is planted to separate positives, and asserts the three reads compute end-to-end with the right shapes.

```python
def _feature_row(pair_id, label, iptm, fold="0"):
    # Minimal row matching extract_mc_features FEATURE_COLUMNS. Geometry/CDR are
    # constant so only ipTM carries signal; interface_plddt blank -> missing flag.
    return {
        "pair_id": pair_id, "label": label, "antigen_cluster": "0", "fold": fold,
        "prediction_present": "1",
        "iptm": str(iptm), "ptm": "0.5", "interface_pae": "15.0",
        "min_interface_pae": "8.0", "interface_plddt": "", "mean_plddt": "70.0",
        "n_interface_residues_binder": "10", "n_interface_residues_target": "12",
        "buried_sasa_proxy_a2": "500.0", "atom_contacts_5a": "40.0",
        "shape_complementarity_proxy": "0.5", "atom_clash_fraction_2a": "0.01",
        "cdr_contact_fraction": "0.6", "cdr1_contact_fraction": "0.2",
        "cdr2_contact_fraction": "0.2", "cdr3_contact_fraction": "0.2",
        "cdr_mapping_ok": "1",
    }


def _write_features(path, rows):
    import csv as _csv
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_analyze_epcam_end_to_end(tmp_path):
    mod = _load()
    rng = __import__("numpy").random.default_rng(0)
    # SAbDab train: 40 positives (high ipTM) + 40 negatives (low ipTM), 5 folds.
    sab = []
    for i in range(40):
        sab.append(_feature_row(f"s{i}__s{i}", "1", 0.6 + 0.1 * rng.random(), fold=str(i % 5)))
        sab.append(_feature_row(f"s{i}__neg0", "0", 0.2 + 0.1 * rng.random(), fold=str(i % 5)))
    sab_path = tmp_path / "sab.csv"; _write_features(sab_path, sab)
    # EpCAM test: 4 designed positives + 8 shuffled negatives.
    epc = []
    for vid in ("10", "25", "14", "16"):
        epc.append(_feature_row(f"epcam-{vid}__epcam", "1", 0.55 + 0.1 * rng.random()))
        for j in range(2):
            epc.append(_feature_row(f"epcam-{vid}__neg{j}", "0", 0.2 + 0.1 * rng.random()))
    epc_path = tmp_path / "epc.csv"; _write_features(epc_path, epc)
    kill = tmp_path / "k.csv"
    kill.write_text("vhh_id,vhh_sequence,label\n10,AA,Good\n25,AA,Good\n14,AA,Bad\n16,AA,Bad\n")

    res = mod.analyze_epcam(
        sab_path, epc_path, kill, l2=1.0, target_precision=0.9, n_boot=200, seed=1
    )
    assert res["primary"]["n"] == 12 and res["primary"]["n_positive"] == 4
    assert res["primary"]["rung0_auroc"] > 0.8  # planted ipTM signal transfers
    assert len(res["primary"]["delta_auroc_r3_minus_r0"]["ci"]) == 2
    assert res["calibration"]["rung3"]["n"] == 12
    assert res["secondary_killing"]["n_functional"] == 2
    assert res["secondary_killing"]["n_nonfunctional"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_analyze_mc_epcam.py::test_analyze_epcam_end_to_end -v`
Expected: FAIL only if a bug exists; if `analyze_epcam` from Task 4 is correct it may already PASS. If it PASSES, that is acceptable (the test still guards the contract) — proceed to add the CLI. If it FAILS, fix `analyze_epcam` until it passes.

- [ ] **Step 3: Add the argparse CLI (mirrors analyze_mc_cross_regime.py)**

Append to `scripts/analyze_mc_epcam.py`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sabdab-features", type=Path, required=True)
    parser.add_argument("--epcam-features", type=Path, required=True)
    parser.add_argument("--killing-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    result = analyze_epcam(
        args.sabdab_features,
        args.epcam_features,
        args.killing_labels,
        l2=args.l2,
        target_precision=args.target_precision,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    p = result["primary"]
    print(f"EpCAM n={p['n']} ({p['n_positive']} pos / {p['n_negative']} neg)")
    print(f"  rung0 (ipTM) AUROC: {round(p['rung0_auroc'], 3)}")
    print(f"  rung3 (full)  AUROC: {round(p['rung3_auroc'], 3)}")
    d = p["delta_auroc_r3_minus_r0"]
    print(f"  Delta(R3-R0): {round(d['point'], 3)} CI {[round(c, 3) for c in d['ci']]}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full analysis test file + lint/type**

Run: `uv run pytest tests/test_analyze_mc_epcam.py -v && uv run ruff check scripts/analyze_mc_epcam.py && uv run mypy src/mirage`
Expected: all PASS; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_mc_epcam.py tests/test_analyze_mc_epcam.py
git commit -m "EpCAM canary analysis: smoke-integration test + CLI"
```

---

## Task 6: Full-suite gate before the GPU campaign

**Files:** none (verification only).

- [ ] **Step 1: Run the complete battery**

Run: `uv run ruff check && uv run ruff format --check && uv run mypy src/mirage && uv run pytest`
Expected: ruff clean, format clean, mypy clean, **all tests pass** (the existing suite + the new EpCAM staging/analysis tests). Fix anything red before touching the cluster.

- [ ] **Step 2: Commit (only if formatting changed)**

```bash
git add -A && git commit -m "EpCAM canary: format/lint pass" || echo "nothing to commit"
```

---

## Task 7: Stage the EpCAM pairs + Protenix inputs (operational)

**Files:** produces `data/staged/epcam/epcam_protenix_pairs.csv` (gitignored) + staged Protenix inputs.

- [ ] **Step 1: Build the pairs CSV**

Run:
```bash
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
uv run python scripts/stage_epcam_protenix_pairs.py \
    --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
    --sabdab-pairs data/staged/sabdab/sabdab_pairs.csv \
    --output data/staged/epcam/epcam_protenix_pairs.csv \
    --k 5
```
Expected: `positives=14 negatives=70 total=84 pool=<N> (dedup max_identity=0.9)`. If it raises `Leakage:` or `Negative pool too small`, STOP and surface — do not work around.

- [ ] **Step 2: Emit per-pair Protenix inputs + the unique-sequence manifest**

Mirror the B1 staging (`stage_protenix_pairs.py` + `ProtenixPosePredictor.stage`). Run:
```bash
uv run python scripts/stage_protenix_pairs.py \
    --pairs data/staged/epcam/epcam_protenix_pairs.csv \
    --staged-root data/staged/epcam/protenix
```
Expected: per-pair input JSONs under `data/staged/epcam/protenix/` + a unique-sequence manifest. Confirm the manifest has ~14 VHH + the EpCAM antigen + the unique drawn wrong antigens (it will reuse already-cached SAbDab antigen MSAs by sequence hash).

- [ ] **Step 3: Sanity-check the staged inputs**

Run: `ls data/staged/epcam/protenix | head` and confirm 84 per-pair entries (or the manifest/`*.json` layout matching B1). Verify no `--use_template true` anywhere (templates must stay off).

- [ ] **Step 4: Commit the staging script outputs that are tracked (none) — note state**

The staged CSV/inputs are gitignored. No commit; record the counts in the session log.

---

## Task 8: Run the Protenix prediction campaign (operational, SLURM)

**Files:** produces predicted complexes under `data/raw/predictions/protenix/epcam/` (gitignored).

Reuse the B1 pipeline verbatim (`precompute_protenix_msa.py` → `predict_protenix.slurm` + `run_protenix_chunk.py`). All Blackwell flags are baked into those scripts — **do not change them** (`LAYERNORM_TYPE=native`, cu128 torch, `--trimul_kernel torch --triatt_kernel torch --enable_fusion false`, `--need_atom_confidence true`, templates off).

- [ ] **Step 1: Precompute MSAs for any new unique sequences**

In the `protenix` conda env (per the B1 HANDOFF):
```bash
export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com
python3 scripts/precompute_protenix_msa.py \
    --unique-manifest data/staged/epcam/protenix/<unique_manifest> \
    --cache-dir data/staged/protenix/msa_cache \
    --work-dir data/staged/protenix/msa_work
```
Expected: a3m files for the new sequences symlinked into the shared cache by sequence hash; already-cached SAbDab antigens skipped. ~14 VHH + a few antigens to fetch.

- [ ] **Step 2: Submit the prediction array**

Submit via `ProtenixPosePredictor.submit()` (the documented path) or the B1 SLURM wrapper, pointing at the EpCAM manifest. 84 pairs is one small array on `b200-mig45` (`--qos=mig`, `--cpus-per-task=8 --mem=56G`). Keep only `sample_0` per the B1 runner.
Expected: 84 predicted complexes under `data/raw/predictions/protenix/epcam/<pair_id>/...sample_0.cif` + `*_full_data_*.json` (PAE present).

- [ ] **Step 3: Install-validation gate (cognate >> shuffled ipTM)**

Before trusting the run, confirm the qualitative signal on the positives:
```bash
uv run python - <<'PY'
import json, glob, statistics
root = "data/raw/predictions/protenix/epcam"
def iptm(pid):
    f = glob.glob(f"{root}/{pid}/**/*summary*.json", recursive=True)
    return json.load(open(f[0])).get("iptm") if f else None
# spot-check a few positive pair_ids vs their negatives
PY
```
Expected: positive (cognate-designed) ipTM medians visibly above shuffled. If positives do not exceed shuffled at all, STOP — the campaign or staging is wrong (do not proceed to scoring).

- [ ] **Step 4: Record campaign state**

Note predicted/total (target 84/84) and any failures in the session log. No git commit (predictions gitignored).

---

## Task 9: Extract features + run the analysis + write the summary

**Files:** produces `data/staged/mc/epcam_features.csv` (gitignored), `results/published/mc_epcam_canary.json` + `mc_epcam_canary_summary.md` (committed).

- [ ] **Step 1: Extract M-C features for the 84 EpCAM pairs**

Run:
```bash
uv run python scripts/extract_mc_features.py \
    --pairs data/staged/epcam/epcam_protenix_pairs.csv \
    --predictions-root data/raw/predictions/protenix/epcam \
    --output data/staged/mc/epcam_features.csv \
    --failed-log data/staged/mc/epcam_features_failed.txt
```
Expected: `built=84 failed=0` (or note any failures). The CSV has the exact `FEATURE_COLUMNS` schema the analysis reads.

- [ ] **Step 2: Run the frozen-transfer analysis**

Run:
```bash
uv run python scripts/analyze_mc_epcam.py \
    --sabdab-features data/staged/mc/sabdab_features.csv \
    --epcam-features data/staged/mc/epcam_features.csv \
    --killing-labels ../abdisc-data/epcam/epcam_killing_labels.csv \
    --output results/published/mc_epcam_canary.json
```
Expected: printed rung0/rung3 AUROC + Δ(R3−R0) CI; `mc_epcam_canary.json` written with `primary`, `calibration`, `secondary_killing`.

- [ ] **Step 3: Write `results/published/mc_epcam_canary_summary.md`**

Author the writeup from the JSON (mirror `mc_indist_summary.md`'s structure + caveat discipline). It MUST include, filled from the JSON (no fabricated numbers):
- Dataset & protocol (14 positives / ~70 shuffled negatives; frozen SAbDab gate, never trained on EpCAM; dedup guard).
- **Primary table:** rung-0 (ipTM) AUROC, rung-3 AUROC, paired Δ(R3−R0) with CI — and whether it clears the in-distribution 0.690 / the M-S 0.496 floor.
- **Calibration:** the frozen P=0.90 threshold's precision/recall/specificity on EpCAM, explicitly compared to the M-S-on-AVIDa recall-0 collapse.
- **Secondary (caveated):** functional-vs-non-functional descriptive separation, flagged N=14 / killing≠binding / hypothesis-generating for AVIDa.
- **Read:** one paragraph on what it means for the project (does ipTM generalize to designed binders; does structure still add nothing).

- [ ] **Step 4: Commit the results**

```bash
git add results/published/mc_epcam_canary.json results/published/mc_epcam_canary_summary.md
git commit -m "M-C EpCAM real-negative canary results: frozen-transfer reads + summary"
```

---

## Task 10: Update the campaign record, wiki, and notes

**Files:** `results/published/mc_campaign_record.md` (append), `../mirage-wiki/wiki/Current State.md`, a new `../mirage-notes/02 - Progress & Records/2026-06-07 - M-C EpCAM real-negative canary.md`.

- [ ] **Step 1: Append an "EpCAM canary" section to the campaign record**

Add the headline numbers (primary AUROC + Δ + calibration verdict) to `results/published/mc_campaign_record.md` under a new dated section.

- [ ] **Step 2: Commit the mirage repo doc**

```bash
git add results/published/mc_campaign_record.md
git commit -m "Record M-C EpCAM canary in the campaign record"
```

- [ ] **Step 3: Update the wiki + notes (separate repos, pull-before / push-after)**

In `mirage-wiki` and `mirage-notes`: `git pull --ff-only`, update `Current State.md` headline status (add the EpCAM canary result + what it relocates), add the dated progress note, then commit + push. **Commits must be Pedram-authored with NO Claude/Anthropic trailer.**

- [ ] **Step 4: Push the mirage code repo**

```bash
git push
```
(If on a feature branch, open a PR titled "M-C EpCAM real-negative canary"; otherwise push `main`. Pedram authors all commits — no Claude trailer.)

---

## Self-Review

**Spec coverage:**
- §2 question 1 (ipTM generalizes?) → Task 9 primary read (rung-0 AUROC). ✓
- §2 question 2 (geometry/CDR add anything here?) → Task 9 primary Δ(R3−R0). ✓
- §2 question 3 (frozen threshold holds?) → Task 9 calibration read. ✓
- §4 positives (14, IDs) → Task 3 `load_epcam_positives` via `epcam_killing` loader. ✓
- §4 negatives (predict-the-shuffled, k=5, SAbDab pool) → Tasks 1–3. ✓
- §4 leakage guards (EpCAM-vs-SAbDab cluster dedup; VHHs not in SAbDab; wrong antigens not EpCAM) → Task 1 dedup pool + Task 3 binder-overlap assertion. ✓
- §5 Protenix config / templates off / MSA cache → Tasks 7–8 (reuse B1, flags unchanged). ✓
- §6 three reads (frozen transfer, no refit) → Tasks 4–5 + Task 9. ✓
- §7 caveats (N=14, killing≠binding) → secondary read caveat string + summary. ✓
- §9 invariants (torch-free, no templates, never train EpCAM, Pedram-authored) → respected throughout. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type/name consistency:** `_FIELDNAMES`, `epcam_antigen_negative_pool`, `build_epcam_pairs`, `load_epcam_positives`, `sabdab_antigen_pool` (staging); `vhh_id_from_pair`, `killing_label_map`, `fit_sabdab_rung`, `analyze_epcam` (analysis) — names used consistently across tasks. Feature-row schema in the smoke test matches `extract_mc_features.FEATURE_COLUMNS`. `MsModel.predict_logit`, `evaluate_frozen_gate`, `paired_delta_bootstrap`, `auroc`, `rung_matrix`/`labels_array`/`folds_array`/`fit_rung_model`/`read_feature_csv` all match their real signatures. ✓

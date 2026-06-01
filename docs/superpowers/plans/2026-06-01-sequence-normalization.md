# Sequence Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize binder/antigen sequences to comparable mature domains before the M-S Tier-S featurization, so the Champloo-frozen gate is trained and evaluated on like-for-like inputs; then re-stage, re-freeze, and re-run the Phase A artifacts.

**Architecture:** A new pure module `mirage/features/normalize.py` exposes two idempotent functions — `normalize_binder` (ANARCI IMGT variable-domain extraction via a mirage-local HMMER, with His-strip cleanup and a safe fallback) and `normalize_antigen` (curated signal-peptide map + terminal His-tag strip). They are called at three sites (the two staging scripts and the orthogonal harness); `sequence_features` stays pure. A small write-boundary fix makes the orthogonal JSON valid when precision is undefined.

**Tech Stack:** Python 3.11, uv, numpy, ANARCI + locally-built HMMER 3.4 (`.tools/hmmer/`), pytest, ruff, mypy strict.

**Prerequisite (already done):** `bash scripts/install_hmmer.sh` has built HMMER into `.tools/hmmer/bin/hmmscan`; the spec and bootstrap script are committed on branch `sequence-normalization`. The avida loader/staging changes and `huggingface_hub` dep are uncommitted in the working tree and get folded into Task 3 / Task 5.

---

### Task 1: Antigen normalization (signal-peptide map + His-strip)

**Files:**
- Modify: `src/mirage/benchmark/targets.py` (add IL-6 signal-peptide constants)
- Create: `src/mirage/features/normalize.py` (antigen half + His-strip helper)
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Add the IL-6 signal-peptide constant to `targets.py`**

Append to `src/mirage/benchmark/targets.py`:

```python
# UniProt P05231 (human IL-6) signal peptide, residues 1-29. Present on the raw
# AVIDa-hIL6 antigen sequences (mature region starts at "VPPGEDSKD..."); absent
# from Champloo's PDB-derived antigens. Stripped during normalization so both
# datasets are featurized on the mature antigen.
IL6_SIGNAL_PEPTIDE: str = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP"

# Known precursor signal-peptide prefixes to strip from antigen sequences.
SIGNAL_PEPTIDES: tuple[str, ...] = (IL6_SIGNAL_PEPTIDE,)
```

- [ ] **Step 2: Write the failing antigen tests**

Create `tests/test_normalize.py`:

```python
from __future__ import annotations

import pytest

from mirage.features import normalize
from mirage.features.normalize import normalize_antigen, normalize_binder

_MATURE = "VPPGEDSKDVAAPHRQ"  # mature-IL6 fragment; not a signal prefix, no His run


def test_normalize_antigen_strips_il6_signal_peptide() -> None:
    raw = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + _MATURE + "HHHHHH"
    assert normalize_antigen(raw) == _MATURE


def test_normalize_antigen_strips_terminal_his() -> None:
    assert normalize_antigen(_MATURE + "HHHHHH") == _MATURE


def test_normalize_antigen_strips_leading_his() -> None:
    assert normalize_antigen("MAHHHHHHSSGVSKGEELFTG") == "SSGVSKGEELFTG"


def test_normalize_antigen_idempotent() -> None:
    raw = "MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + _MATURE + "HHHHHH"
    once = normalize_antigen(raw)
    assert normalize_antigen(once) == once
```

- [ ] **Step 3: Run the antigen tests to verify they fail**

Run: `uv run pytest tests/test_normalize.py -k antigen -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mirage.features.normalize'`

- [ ] **Step 4: Implement the antigen half of `normalize.py`**

Create `src/mirage/features/normalize.py`:

```python
"""Normalize binder/antigen sequences to comparable mature domains before the
Tier-S featurization. ``normalize_binder`` extracts the antibody variable domain
via ANARCI (IMGT) using a mirage-local HMMER; ``normalize_antigen`` strips known
signal peptides and terminal His-tags. Both are idempotent."""

from __future__ import annotations

import re

from mirage.benchmark.targets import SIGNAL_PEPTIDES

_HIS_LEAD = re.compile(r"^[MA]{0,3}H{5,}")
_HIS_TAIL = re.compile(r"H{5,}$")


def _strip_his_tags(seq: str) -> str:
    """Remove an unambiguous terminal His-run (>=5) at either end. Conservative:
    the leading form tolerates a short Met/Ala cloning prefix; no internal cuts."""
    seq = _HIS_LEAD.sub("", seq)
    seq = _HIS_TAIL.sub("", seq)
    return seq


def normalize_antigen(seq: str) -> str:
    """Strip a known precursor signal peptide (if present) then terminal His-tags."""
    seq = seq.strip().upper()
    for signal in SIGNAL_PEPTIDES:
        if seq.startswith(signal):
            seq = seq[len(signal):]
            break
    return _strip_his_tags(seq)
```

- [ ] **Step 5: Run the antigen tests to verify they pass**

Run: `uv run pytest tests/test_normalize.py -k antigen -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/mirage/benchmark/targets.py src/mirage/features/normalize.py tests/test_normalize.py
git commit -m "Add antigen normalization: signal-peptide map + His-tag strip"
```

---

### Task 2: Binder normalization (ANARCI variable-domain extraction)

**Files:**
- Modify: `src/mirage/features/normalize.py` (add binder half + HMMER resolver)
- Test: `tests/test_normalize.py` (add binder tests)

- [ ] **Step 1: Write the failing binder tests**

Append to `tests/test_normalize.py`:

```python
_LEADER = "MKYLLPTAAAGLLLLAAQPAMA"
_VHH = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKG"
    "RFTISRDNANNTVYLQMNSLKPEDTAVYACAADLLYHPGSWNDYWGQGTQVTVSS"
)
_HAS_HMMER = normalize._resolve_hmmer_bin() is not None
requires_hmmer = pytest.mark.skipif(not _HAS_HMMER, reason="HMMER/hmmscan unavailable")


def test_normalize_binder_falls_back_without_hmmer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the dependency-free path; fallback is His-strip only.
    monkeypatch.setattr(normalize, "_resolve_hmmer_bin", lambda: None)
    normalize._anarci_domain.cache_clear()
    assert normalize_binder("RANDOMSEQ" + "HHHHHH") == "RANDOMSEQ"


@requires_hmmer
def test_normalize_binder_extracts_variable_domain() -> None:
    dom = normalize_binder(_LEADER + _VHH + "HHHHHH")
    assert dom.startswith("QVQL")
    assert dom.endswith("VTVSS")
    assert "MKYLL" not in dom
    assert "HHHHHH" not in dom


@requires_hmmer
def test_normalize_binder_idempotent() -> None:
    once = normalize_binder(_LEADER + _VHH + "HHHHHH")
    assert normalize_binder(once) == once
```

- [ ] **Step 2: Run the binder tests to verify they fail**

Run: `uv run pytest tests/test_normalize.py -k binder -v`
Expected: FAIL — `AttributeError: module 'mirage.features.normalize' has no attribute '_resolve_hmmer_bin'` (collection error)

- [ ] **Step 3: Implement the binder half of `normalize.py`**

Add to the imports at the top of `src/mirage/features/normalize.py`:

```python
import functools
import os
import shutil
from pathlib import Path
```

Append after `normalize_antigen`:

```python
def _resolve_hmmer_bin() -> str | None:
    """Locate a HMMER bin dir containing hmmscan: env var, then PATH, then the
    repo-local .tools/hmmer/bin built by scripts/install_hmmer.sh."""
    env = os.environ.get("MIRAGE_HMMER_BIN")
    if env and (Path(env) / "hmmscan").exists():
        return env
    on_path = shutil.which("hmmscan")
    if on_path:
        return str(Path(on_path).parent)
    repo_root = Path(__file__).resolve().parents[3]
    default = repo_root / ".tools" / "hmmer" / "bin"
    if (default / "hmmscan").exists():
        return str(default)
    return None


@functools.lru_cache(maxsize=None)
def _anarci_domain(seq: str) -> str | None:
    """Return the IMGT variable-domain substring of ``seq``, or None if ANARCI /
    HMMER is unavailable or no antibody domain is found. Memoized over unique
    sequences (AVIDa has few unique VHHs across its 573k rows)."""
    hmmer_bin = _resolve_hmmer_bin()
    if hmmer_bin is None:
        return None
    try:
        from anarci import run_anarci  # type: ignore[import-untyped]

        result = run_anarci([("q", seq)], scheme="imgt", hmmerpath=hmmer_bin)
    except Exception:
        return None
    details = result[2][0]
    if not details:
        return None
    domain = details[0]
    start = int(domain["query_start"])
    end = int(domain["query_end"])
    return seq[start : end + 1]


def normalize_binder(seq: str) -> str:
    """Reduce an antibody chain to its IMGT variable domain (drops leader peptide,
    His-tags, framing residues). Falls back to His-strip when ANARCI/HMMER is
    unavailable or no domain is found."""
    seq = seq.strip().upper()
    domain = _anarci_domain(seq)
    if domain is None:
        return _strip_his_tags(seq)
    return _HIS_TAIL.sub("", domain)
```

- [ ] **Step 4: Run the binder tests to verify they pass**

Run: `uv run pytest tests/test_normalize.py -k binder -v`
Expected: PASS — fallback test passes; the two `requires_hmmer` tests pass (HMMER is built locally) or SKIP where HMMER is absent.

- [ ] **Step 5: Run the full normalize suite + lint/type**

Run: `uv run pytest tests/test_normalize.py -v && uv run ruff check src/mirage/features/normalize.py && uv run mypy src/mirage/features/normalize.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/mirage/features/normalize.py tests/test_normalize.py
git commit -m "Add binder normalization: ANARCI IMGT domain via mirage-local HMMER"
```

---

### Task 3: Wire normalization into staging scripts + orthogonal harness

**Files:**
- Modify: `scripts/stage_avida.py` (normalize in `build_rows`)
- Modify: `scripts/stage_champloo_features.py` (normalize in `build_feature_rows`)
- Modify: `src/mirage/eval/orthogonal.py` (normalize in `features_for_examples`)
- Test: `tests/test_normalize_wiring.py` (new) + update `tests/test_avida_loader.py`

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_normalize_wiring.py`:

```python
from __future__ import annotations

from mirage.eval.orthogonal import features_for_examples
from mirage.scorers.base import BenchmarkExample


def test_features_for_examples_normalizes_antigen_signal_peptide() -> None:
    # Two examples identical except one antigen carries the IL-6 signal peptide;
    # after normalization their target features must be identical.
    mature = "VPPGEDSKDVAAPHRQ"
    ex_raw = BenchmarkExample(
        id="raw", label="BIND", binder_chains=("QVQL",), binder_format="vhh",
        target_chains=("MNSFSTSAFGPVAFSLGLLLVLPAAFPAP" + mature,), target_name="IL6",
    )
    ex_mature = BenchmarkExample(
        id="mat", label="BIND", binder_chains=("QVQL",), binder_format="vhh",
        target_chains=(mature,), target_name="IL6",
    )
    x, _y, _names = features_for_examples([ex_raw, ex_mature], positive_label="BIND")
    assert list(x[0]) == list(x[1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_normalize_wiring.py -v`
Expected: FAIL — feature rows differ (signal peptide not yet stripped).

- [ ] **Step 3: Wire `features_for_examples` in `src/mirage/eval/orthogonal.py`**

Add the import near the other `mirage.features` import:

```python
from mirage.features.normalize import normalize_antigen, normalize_binder
```

In `features_for_examples`, replace the line:

```python
        feats = sequence_features(ex.binder_chains[0], ex.target_chains[0])
```

with:

```python
        feats = sequence_features(
            normalize_binder(ex.binder_chains[0]),
            normalize_antigen(ex.target_chains[0]),
        )
```

- [ ] **Step 4: Run the wiring test to verify it passes**

Run: `uv run pytest tests/test_normalize_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Wire `scripts/stage_avida.py`**

Add the import below the existing imports:

```python
from mirage.features.normalize import normalize_antigen, normalize_binder
```

In `build_rows`, replace:

```python
        seq = rec.get("VHH_sequence", "")
        antigen_seq = antigens.get(ag_label, "")
```

with:

```python
        seq = normalize_binder(rec.get("VHH_sequence", ""))
        antigen_seq = normalize_antigen(antigens.get(ag_label, ""))
```

- [ ] **Step 6: Wire `scripts/stage_champloo_features.py`**

Add the import below the existing `from mirage.features.sequence import ...` line:

```python
from mirage.features.normalize import normalize_antigen, normalize_binder
```

In `build_feature_rows`, replace:

```python
        binder_seq = vhh["vhh_sequence"]
        target_seq = ant["antigen_sequence"]
```

with:

```python
        binder_seq = normalize_binder(vhh["vhh_sequence"])
        target_seq = normalize_antigen(ant["antigen_sequence"])
```

- [ ] **Step 7: Update `tests/test_avida_loader.py` for normalization-aware staging**

In `test_stage_avida_joins_antigen_sequences`, the toy antigen `MNSFSTSAFGPVAFSLGLLLVLPAAFPAP` is exactly the IL-6 signal peptide and would normalize to empty. Replace the test body's antigen and assertions:

```python
def test_stage_avida_joins_antigen_sequences() -> None:
    stage = _load_script("stage_avida.py")
    records = [
        {"VHH_sequence": "QVQL", "Ag_label": "IL6", "label": "1"},
        {"VHH_sequence": "EVQL", "Ag_label": "IL6", "label": "0"},
    ]
    # mature-domain antigen (no signal prefix, no His run): passes through unchanged
    antigens = {"IL6": "VPPGEDSKDVAAPHRQ"}
    rows = stage.build_rows(records, antigens)
    assert rows[0]["label"] == "1"
    assert rows[0]["antigen_sequence"] == "VPPGEDSKDVAAPHRQ"
    assert rows[1]["label"] == "0"
```

(`test_antigen_map_accepts_ag_sequence_column` and `test_avida_loader_yields_examples` are unaffected — leave them.)

- [ ] **Step 8: Run the affected suites**

Run: `uv run pytest tests/test_normalize_wiring.py tests/test_avida_loader.py -v`
Expected: PASS (toy VHHs hit the ANARCI no-domain fallback and pass through).

- [ ] **Step 9: Commit**

```bash
git add scripts/stage_avida.py scripts/stage_champloo_features.py src/mirage/eval/orthogonal.py tests/test_normalize_wiring.py tests/test_avida_loader.py pyproject.toml uv.lock
git commit -m "Normalize sequences at staging + orthogonal harness; add huggingface_hub dep"
```

---

### Task 4: Orthogonal reporting robustness (undefined precision -> null)

**Files:**
- Modify: `scripts/analyze_ms_orthogonal.py` (sanitize NaN at the write boundary)
- Test: `tests/test_orthogonal_report.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_orthogonal_report.py`:

```python
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any


def _load_script(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_safe_replaces_nan_with_none() -> None:
    mod = _load_script("analyze_ms_orthogonal.py")
    safe = mod._json_safe({"precision": math.nan, "recall": 0.0, "ci": [math.nan, 1.0]})
    # Must round-trip through strict JSON (NaN would be rejected).
    text = json.dumps(safe, allow_nan=False)
    back = json.loads(text)
    assert back["precision"] is None
    assert back["recall"] == 0.0
    assert back["ci"] == [None, 1.0]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_orthogonal_report.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_json_safe'`

- [ ] **Step 3: Implement `_json_safe` and apply it at the write boundary**

In `scripts/analyze_ms_orthogonal.py`, add `import math` to the imports, then add this helper above `main`:

```python
def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats (NaN/inf) with None so the table is valid strict
    JSON. Precision is undefined (None) when the gate predicts no positives."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj
```

Replace the write + print block at the end of `main`:

```python
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    for regime, res in table["regimes"].items():
        m = res["metrics"]
        print(
            f"{regime}: n={res['n']} precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} specificity={m['specificity']:.3f}"
        )
    print(f"Wrote {args.output}")
    return 0
```

with:

```python
    safe = _json_safe(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe, indent=2))

    def _fmt(v: Any) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    for regime, res in safe["regimes"].items():
        m = res["metrics"]
        print(
            f"{regime}: n={res['n']} precision={_fmt(m['precision'])} "
            f"recall={_fmt(m['recall'])} specificity={_fmt(m['specificity'])}"
        )
    print(f"Wrote {args.output}")
    return 0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_orthogonal_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_ms_orthogonal.py tests/test_orthogonal_report.py
git commit -m "Serialize undefined precision as null in the orthogonal table"
```

---

### Task 5: Re-stage, re-freeze, re-run, re-fill the Phase A artifacts

**Files:**
- Regenerate (gitignored): `data/staged/avida/avida_staged.csv`, `data/staged/champloo/champloo_features_af3.csv`
- Regenerate (tracked): `results/published/ms_model_af3.json`, `results/published/mirage_ms_indist_af3.json`, `results/published/mirage_ms_orthogonal.json`
- Modify: `results/published/mirage_phase_a_summary.md`

- [ ] **Step 1: Confirm HMMER is present (prerequisite)**

Run: `test -x .tools/hmmer/bin/hmmscan && echo OK || bash scripts/install_hmmer.sh`
Expected: `OK`

- [ ] **Step 2: Re-stage AVIDa with normalization**

Run:
```bash
uv run python scripts/stage_avida.py \
  --records data/raw/avida/AVIDa-hIL6.csv \
  --antigens data/raw/avida/antigen_sequences.csv \
  --output data/staged/avida/avida_staged.csv
```
Expected: `Wrote <N> AVIDa rows ...`. Spot-check a row: the VHH no longer starts with `MKYLL`, the antigen starts with `VPPGEDSKD`, neither ends in `HHHHHH`:
```bash
uv run python -c "import csv;r=next(csv.DictReader(open('data/staged/avida/avida_staged.csv')));print('vhh',r['vhh_sequence'][:8],'| ag',r['antigen_sequence'][:9],'| vhh_his',r['vhh_sequence'].endswith('HHHHHH'),'| ag_his',r['antigen_sequence'].endswith('HHHHHH'))"
```
Expected: `vhh QVQLQESG | ag VPPGEDSKD | vhh_his False | ag_his False`

- [ ] **Step 3: Re-stage Champloo features with normalization**

Run:
```bash
uv run python scripts/stage_champloo_features.py \
  --pairs data/staged/champloo/champloo_pairs_af3.csv \
  --supp ../abdisc-data/champloo/Supplementary_Table_1_*.csv \
  --output data/staged/champloo/champloo_features_af3.csv
```
Expected: writes the feature CSV. Sanity check binder_length dropped toward the mature ~123 (His tails removed):
```bash
uv run python -c "import csv,statistics as s;rows=list(csv.DictReader(open('data/staged/champloo/champloo_features_af3.csv')));print('mean binder_length',round(s.mean(float(r['binder_length']) for r in rows),1))"
```
Expected: mean binder_length in the ~118-124 range (was ~126 with His tails).

- [ ] **Step 4: Re-train / re-freeze the in-distribution model**

Run:
```bash
uv run python scripts/analyze_ms_indist.py \
  --features data/staged/champloo/champloo_features_af3.csv \
  --model-out results/published/ms_model_af3.json \
  --output results/published/mirage_ms_indist_af3.json
```
Expected: prints the in-dist AUROC line. Confirm it is still ~0.3-0.4 (no binding signal); if it moved materially (>0.55), STOP and report — that contradicts the spec's expectation and needs investigation before continuing.

- [ ] **Step 5: Re-run the orthogonal gate on AVIDa**

Run:
```bash
uv run python scripts/analyze_ms_orthogonal.py \
  --model results/published/ms_model_af3.json \
  --avida-csv data/staged/avida/avida_staged.csv \
  --output results/published/mirage_ms_orthogonal.json
```
Expected: prints `avida: n=... precision=... recall=... specificity=...`. Confirm the JSON is now strict-valid:
```bash
uv run python -c "import json;json.loads(open('results/published/mirage_ms_orthogonal.json').read());print('valid strict JSON')"
```
Expected: `valid strict JSON` (no bare NaN).

- [ ] **Step 6: Fill the Phase A summary**

Read `results/published/mirage_phase_a_summary.md`. Fill the in-distribution and orthogonal rows with the actual numbers from `mirage_ms_indist_af3.json` and `mirage_ms_orthogonal.json` produced above. Do NOT fabricate — copy values from the JSON. Add a one-line note that sequences are normalized to mature domains (ANARCI variable domain for binders; signal-peptide/His-tag strip for antigens) per `docs/superpowers/specs/2026-06-01-sequence-normalization-design.md`. Leave the labeled-EpCAM row marked blocked (no labels file).

- [ ] **Step 7: Commit the regenerated artifacts**

```bash
git add results/published/ms_model_af3.json results/published/mirage_ms_indist_af3.json results/published/mirage_ms_orthogonal.json results/published/mirage_phase_a_summary.md
git commit -m "Re-stage/re-freeze/re-run Phase A on normalized mature-domain sequences"
```

---

### Task 6: Full verification battery

**Files:** none (verification only)

- [ ] **Step 1: Run the full battery**

Run:
```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/mirage && uv run pytest
```
Expected: ruff clean, format clean, mypy clean, all tests pass (binder ANARCI tests run locally since HMMER is built; they SKIP only where HMMER is absent).

- [ ] **Step 2: Confirm the branch state**

Run: `git status && git log --oneline origin/main..HEAD`
Expected: clean working tree; the commit list shows the spec, the bootstrap, and Tasks 1-5 on `sequence-normalization`, ahead of `origin/main`.

- [ ] **Step 3: Report to the user**

Summarize: the normalized in-dist AUROC vs the previous 0.362, the new AVIDa orthogonal metrics vs the previous all-negative collapse, and confirm labeled-EpCAM remains blocked. Ask whether to push the branch / open a PR (do not push without the user's go-ahead).
```

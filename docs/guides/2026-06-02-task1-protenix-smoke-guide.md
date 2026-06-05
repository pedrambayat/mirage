# Task 1 Guide — Protenix install + single-pair smoke + schema capture

**You are the operator for this task.** It's the M-C feasibility gate: prove Protenix
runs on PARCC and capture the real input/output schema that Tasks 2–8 are written
against. Everything downstream (the confidence parser, the chain resolver, the MSA
pre-compute) depends on the three things you document here.

**Plan reference:** `docs/superpowers/plans/2026-06-02-mirage-mc-phase-b1.md`, Task 1.
**Spec:** `docs/superpowers/specs/2026-06-02-mirage-mc-structure-track-design.md`.

---

## What "done" looks like (the gate)

You may proceed to Task 2 only when **all** of these hold:

1. ✅ `protenix pred` (alias: `protenix predict`) runs to completion on the `3OGO`
   GFP-nanobody pair on a B200 and writes a structure + confidence JSON.
2. ✅ The cognate **ipTM is high** (≳0.7 — GFP nanobody is a real, easy binder; record the exact value).
3. ✅ `docs/datasets/protenix-output-schema.md` records the **three reconciliation points**:
   - **(A) output structure** — file path/format + which chain ID got the binder vs the antigen (and whether Protenix preserved or reordered input order);
   - **(B) confidence JSON** — filename(s) + exact keys for `iptm`, `ptm`, the PAE matrix, and per-residue/atom pLDDT;
   - **(C) MSA wiring** — the input-JSON field that points a chain at a precomputed `.a3m`, and the CLI flag that disables the built-in MSA search.
4. ✅ A trimmed real output is committed at `tests/fixtures/protenix/3OGO__3OGO/` (the parser's test oracle).

If ipTM is *not* high, stop and debug before doing anything else — a low cognate ipTM means the install, the MSA, or the input is wrong, and the whole campaign would be built on sand.

---

## Step 0 — Context check

```bash
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
git branch --show-current          # expect: mc-structure-track
mkdir -p data/raw/predictions/protenix/_smoke docs/datasets tests/fixtures/protenix
```

---

## Step 1 — Install Protenix in its own conda env

Protenix is ByteDance's open AF3-class predictor (`github.com/bytedance/Protenix`).
Keep it **out** of the mirage uv env — it pulls torch-CUDA.

```bash
conda create -y -n protenix python=3.11
conda activate protenix
pip install --upgrade protenix --index-url https://pypi.org/simple
python -c "import protenix; print('protenix', getattr(protenix,'__version__','installed'))"
which protenix                      # confirm the CLI is on PATH
protenix pred -h                    # current upstream primary inference command
```

If `pip install protenix` fails or the CLI name differs in the released version,
follow the install section of the **Protenix README** (clone + `pip install -e .`),
and **record the exact commands you used** in the schema doc. Model weights download
on first `predict` (or via the repo's weight-download script) — let the first run
fetch them; it may take a few minutes.

> **Templates / leakage:** Protenix's open release does not take user structural
> templates, so the "no crystal template" leakage guard is satisfied by default.
> Just confirm you never pass a template flag/field. Note this in the schema doc.

---

## Step 2 — Build the single-pair input (real 3OGO sequences)

This pulls the GFP-nanobody sequences straight from the Champloo table (no manual
transcription) and writes a Protenix input JSON. **Binder chain first, antigen
second** — the project convention. Run it in the **mirage uv env** (it only reads a CSV):

```bash
conda deactivate     # back to base/uv shell; this step needs no GPU
uv run python - <<'PY'
import csv, json, pathlib
CSV = pathlib.Path("../abdisc-data/champloo/Supplementary_Table_1_final_experimental_vhh_ag_systems.csv")
row = next(r for r in csv.DictReader(CSV.open()) if r["pdb_id"].strip() == "3OGO")
vhh, ag = row["vhh_sequence"].strip(), row["antigen_sequence"].strip()
print(f"VHH len={len(vhh)}  GFP len={len(ag)}")

# Protenix input JSON. CONFIRM this shape against the Protenix README quickstart
# example — field names (proteinChain / sequence / count) may differ by version.
job = {
    "name": "3OGO__3OGO",
    "sequences": [
        {"proteinChain": {"sequence": vhh, "count": 1}},   # chain 1 = binder
        {"proteinChain": {"sequence": ag,  "count": 1}},   # chain 2 = antigen
    ],
}
out = pathlib.Path("data/raw/predictions/protenix/_smoke/3OGO.json")
out.write_text(json.dumps([job], indent=2))
print("wrote", out)
PY
```

If Protenix's example JSON uses a different structure, **edit the dict to match it**
and note the correct format under reconciliation point (C) in the schema doc.

---

## Step 3 — Run Protenix on a B200 (MSA server on, for this one pair)

For the *single* smoke pair, let Protenix fetch/build the MSA automatically
(`--use_msa true`, the current default for the base models). MSA pre-compute/dedup
matters only for the 3,234-pair campaign (Tasks 5–6), not here.

```bash
srun -A dbgoodma-goodman-laboratory -p b200-mig45 --gres=gpu:1 --time=01:00:00 --pty bash
# --- inside the interactive shell ---
conda activate protenix
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
protenix pred \
  --input data/raw/predictions/protenix/_smoke/3OGO.json \
  --out_dir data/raw/predictions/protenix/_smoke/3OGO_out \
  --seeds 0 \
  --model_name protenix_base_default_v1.0.0 \
  --use_msa true \
  --use_template false \
  --use_default_params true
# (confirm the exact flag names against `protenix pred --help`)
```

Expected: it builds an MSA, runs inference, and writes structures + confidence JSONs
under `3OGO_out/`. Wall time on a B200 for 372 residues should be small (minutes),
plus MSA time.

**Known gotchas:**
- *Compute node has no outbound network* → the automatic MSA search may fail. Fixes:
  run `protenix prep`/`protenix msa` from a node/login shell that *does* have egress,
  or precompute the two MSAs first and pass them in (preview of Task 6). Test egress
  inside the srun shell: `curl -sI https://pypi.org >/dev/null && echo NET-OK || echo NO-NET`.
- *Weights still downloading* → first run is slow; let it finish, re-run is fast.
- *OOM on the 45 GB MIG slice* → unlikely at 372 residues; if it happens, use
  `-p b200-mig90` or a full `dgx-b200` GPU.

Keep the `srun` shell open for Step 4, or exit and inspect from a normal shell.

---

## Step 4 — Inspect the output and capture the schema

Walk the output tree and dump the JSON keys so you can fill the schema doc fast:

```bash
uv run python - <<'PY'
import json, pathlib
root = pathlib.Path("data/raw/predictions/protenix/_smoke/3OGO_out")
print("=== files ===")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(root)}   ({p.stat().st_size} B)")
print("\n=== JSON top-level keys ===")
for p in sorted(root.rglob("*.json")):
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(p.name, "– not JSON:", e); continue
    keys = list(data)[:25] if isinstance(data, dict) else f"<{type(data).__name__}>"
    print(f"\n{p.relative_to(root)}\n  keys: {keys}")
    if isinstance(data, dict):
        for k in ("iptm", "ptm", "ranking_score", "plddt", "pae", "chain_pair_iptm",
                  "atom_plddts", "token_chain_ids", "contact_probs"):
            if k in data:
                v = data[k]
                shape = (len(v) if isinstance(v, list) else v)
                print(f"    {k!r}: {type(v).__name__} -> {str(shape)[:60]}")
PY
```

Then identify the **chain mapping** in the predicted structure. For a CIF:

```bash
uv run python - <<'PY'
import pathlib
root = pathlib.Path("data/raw/predictions/protenix/_smoke/3OGO_out")
cif = next(root.rglob("*.cif"), None) or next(root.rglob("*.pdb"))
print("structure:", cif.name)
# crude chain + length read so you can tell which chain is the 123-aa VHH vs 249-aa GFP
import gemmi  # protenix env has it; else parse manually
st = gemmi.read_structure(str(cif))
for ch in st[0]:
    print(f"  chain {ch.name}: {len(ch)} residues")
PY
```
The chain with ~123 residues is the **binder**; ~249 is the **antigen**. Record which
chain ID each got, and whether that matches input order (binder-first) or was
reordered — this is *why* Task 2 resolves chains by sequence instead of position.

---

## Step 5 — Write the schema doc

Create `docs/datasets/protenix-output-schema.md` and fill in the real values:

```markdown
# Protenix output schema (captured 2026-06-02, smoke pair 3OGO GFP-nanobody)

protenix version: <from Step 1>
install command:  <pip install protenix | git + pip -e .>
predict command:  protenix pred --input ... --out_dir ... --seeds 0 --model_name protenix_base_default_v1.0.0 --use_msa true --use_template false
templates: none (open Protenix takes no user templates; leakage guard satisfied)

## (A) Output structure
- top-ranked structure file: <relative path, e.g. .../seed_0/predictions/3OGO__3OGO_sample_0.cif>
- format: <cif|pdb>
- chain IDs: binder(123aa) = <?>, antigen(249aa) = <?>
- input order preserved? <yes|NO — reordered to ...>
- pLDDT in B-factor column? <yes|no>

## (B) Confidence JSON
- summary file glob: <e.g. *summary_confidence_sample_*.json>
- iptm key: <iptm>            ptm key: <ptm>
- PAE: file <...>  key <pae>  shape <NxN over tokens? residues?>
- per-residue/atom pLDDT: file <...> key <atom_plddts|plddt>  (atom- or residue-level?)
- chain-pair iptm key (if present): <chain_pair_iptm>
- cognate ipTM observed: <0.??>

## (C) MSA wiring (for Tasks 5–6)
- precomputed-MSA input field per chain: `proteinChain.msa.precomputed_msa_dir`
  plus `proteinChain.msa.pairing_db` (current example uses `uniref100`)
- flag to DISABLE built-in MSA search/use during prediction: `--use_msa false`
- MSA precompute commands: `protenix prep --input ... --out_dir ...` for full
  prep, `protenix mt --input ... --out_dir ...` for protein MSA + template, or
  `protenix msa --input ... --out_dir ... --msa_server_mode protenix` for MSA only
```

Anything you couldn't determine, write `UNKNOWN — revisit` rather than guessing; an
explicit gap is better than a wrong constant baked into the Task 2 parser.

---

## Step 6 — Capture the test fixture

Copy a **trimmed** real output into the fixture dir (the Task 3 parser is TDD'd
against it). Keep the small summary JSON as-is; if a PAE file is large, downsample it
and note the downsampling in the schema doc.

```bash
mkdir -p tests/fixtures/protenix/3OGO__3OGO
# copy the summary confidence JSON (small); adjust the source path to the real one:
cp data/raw/predictions/protenix/_smoke/3OGO_out/**/[!.]*summary_confidence*sample*0*.json \
   tests/fixtures/protenix/3OGO__3OGO/ 2>/dev/null || echo "adjust the glob to the real filename"
# also copy the top-ranked structure (or a header-only trimmed copy if it's big)
```
The fixture must be enough for `ProtenixConfidenceScorer` to read `iptm`, `ptm`, PAE,
and pLDDT for this one example. If the real PAE/structure files are megabytes, store a
reduced version and record exactly what you reduced.

---

## Step 7 — Sanity-check the gate + commit

```bash
# eyeball the cognate ipTM you recorded — must be high (≳0.7)
git add docs/datasets/protenix-output-schema.md tests/fixtures/protenix/
git commit -m "Document Protenix input+output schema (MSA field, chain order) + 3OGO fixture"
```

Do **not** commit `data/raw/predictions/` (gitignored, large). Only the schema doc +
the trimmed fixture.

---

## Final checklist (the gate)

- [ ] `protenix` installed; `protenix pred` completes on 3OGO on a B200
- [ ] cognate ipTM recorded and **high (≳0.7)**
- [ ] schema doc records (A) output structure + chain IDs/order, (B) confidence keys, (C) MSA field + disable flag
- [ ] trimmed fixture committed at `tests/fixtures/protenix/3OGO__3OGO/`
- [ ] schema doc + fixture committed (no large blobs)

When these are all checked, ping me and I'll drive Tasks 2–8 (the pure-Python TDD
units) via subagents against your captured schema + fixture.

---

## If you get stuck

- **ipTM is low (<0.5) for GFP nanobody** → almost certainly the MSA didn't build
  (no network on the node) or the input JSON chains are malformed. Verify the MSA
  step produced a non-trivial alignment, and that the two `proteinChain` sequences
  are the 123-aa VHH and 249-aa GFP.
- **Can't tell which chain is which** → use the residue-count read in Step 4; 123 vs
  249 is unambiguous for this pair.
- **Protenix CLI/JSON differs from this guide** → trust the Protenix README/`--help`
  over this guide and record the real shapes in the schema doc; that recording *is*
  the deliverable.
- **Stuck >30 min on install/run** → capture the error and hand it back to me; a
  broken Protenix install is itself a finding that changes the predictor decision.
```

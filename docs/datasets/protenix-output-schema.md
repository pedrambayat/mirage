# Protenix output schema

Captured 2026-06-03 from the smoke pair **3OGO** (GFP / anti-GFP nanobody) on a PARCC B200.
This is the input/output contract that the M-C structure-track Tasks 2–8 are written against.

```
protenix version: 2.0.0           (conda env `protenix`, python 3.11)
install:          pip install protenix      THEN swap torch to the cu128 build (see Install caveats)
predict command:  protenix pred --input <job.json> --out_dir <dir> --seeds 0 \
                    --model_name protenix_base_default_v1.0.0 --use_msa true \
                    --msa_server_mode colabfold --use_template false --use_default_params true \
                    --trimul_kernel torch --triatt_kernel torch --enable_fusion false
templates:        none (open Protenix takes no user structural templates → crystal-template
                  leakage guard satisfied by default; never pass --use_template true / templatesPath)
```

**Gate result (cognate ipTM):** `iptm = 0.938` across all 5 samples (ptm 0.928, plddt 91.8,
has_clash false). Well above the ≳0.7 bar — Protenix recovers this real binder strongly, so
the install + MSA + inference path is validated.

---

## Install caveats (REQUIRED on PARCC B200 — without these it does not run)

1. **`export LAYERNORM_TYPE=native`** — the CLI JIT-compiles a fused-LayerNorm CUDA kernel
   at import time, which needs `nvcc` (not present in the env). `native` selects the pure-torch
   OpenFold LayerNorm. Without it even `protenix pred -h` crashes (`CUDA_HOME not set`).
2. **torch must be the cu128 build.** The released dep pins `torch==2.7.1+cu126`, which has
   **no Blackwell (sm_100) kernels** — every CUDA op dies with "no kernel image is available
   for execution on the device". Fix:
   `pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
   torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128` (pin the explicit `+cu128` local version
   or pip treats `==2.7.1` as already satisfied and no-ops).
3. **Use the torch triangle kernels, not cuequivariance.** With the default
   `--trimul_kernel cuequivariance --triatt_kernel cuequivariance` (and `--enable_fusion true`),
   inference **hangs** on Blackwell right after featurization (observed ≥25 min with no progress,
   on both a MIG slice and a full B200; cuequivariance also warns `pynvml: Not Supported`).
   With `--trimul_kernel torch --triatt_kernel torch --enable_fusion false` the same pair
   completes in **~19 s**. Use the torch kernels until cuequivariance/Blackwell is sorted.
4. Weights are cached at `PROTENIX_ROOT_DIR=/vast/projects/dbgoodma/goodman-laboratory/pbayat/.protenix`
   (checkpoint + CCD, ~2.1 GB; reusable for the campaign). Default would be `$HOME`.

---

## (A) Output structure

- Per seed, **5 samples** are written (diffusion `--sample 5`, ranked):
  `3OGO__3OGO/seed_0/predictions/3OGO__3OGO_sample_{0..4}.cif`
- Format: **mmCIF** (`.cif`). One model per file. ~295–300 KB each.
- **Chain IDs:** `A` = 123-residue **VHH (binder)**, `B` = 249-residue **GFP (antigen)**.
- **Input order PRESERVED**: input JSON was binder-first (id `A`) / antigen-second (id `B`),
  and the structure keeps `A`=binder, `B`=antigen. (Task 2 should still resolve chains by
  sequence, not position, for robustness across pairs — but for this pair order held.)
- **pLDDT is in the CIF B-factor column** (per-atom, 0–100 scale; e.g. 72–83 here).
- Top-ranked sample = `sample_0` (dumper sorts by ranking_score).

## (B) Confidence JSON

Two JSON families per sample, under the same `predictions/` dir:

### Summary confidence (always written) — `3OGO__3OGO_summary_confidence_sample_{N}.json` (~1.25 KB)
Scalar + per-chain confidence. Keys:
- `iptm` (float) — **the headline inter-chain ipTM** (0.938). `ptm` (float) = 0.928.
- `plddt` (float, **0–100** scale, global mean) ; `chain_plddt` [N_chain] ; `chain_pair_plddt` [N_chain,N_chain]
- `chain_iptm` [N_chain] ; `chain_ptm` [N_chain] ;
  `chain_pair_iptm` [N_chain,N_chain] = `[[0, 0.938],[0.938, 0]]` ;
  `chain_pair_iptm_global` [N_chain,N_chain]
- `gpde` / `chain_gpde` [N_chain] / `chain_pair_gpde` [N_chain,N_chain]
- `has_clash` (bool) ; `disorder` (float) ; `ranking_score` (float) ; `num_recycles` (int=10)
- **No PAE matrix and no per-residue arrays in the summary file.**

### Full data (only with `--need_atom_confidence true`) — `3OGO__3OGO_full_data_sample_{N}.json` (~6.2 MB)
This is where **PAE** lives. Keys (N_token=372, N_atom=2951 for this pair):
- `token_pair_pae`  — **PAE matrix**, shape **(N_token, N_token)** float (Å; ~0.3–29 here). Token-level, NOT residue- or atom-level.
- `token_pair_pde`  — predicted distance error, (N_token, N_token).
- `contact_probs`   — (N_token, N_token), 0–1.
- `atom_plddt`      — per-**atom** pLDDT, (N_atom,), **0–1 scale** (note: summary `plddt` is 0–100).
- `token_asym_id`   — (N_token,) int chain index per token (0 = chain A/VHH, 1 = chain B/GFP). Use to slice inter-chain PAE blocks.
- `atom_to_token_idx` — (N_atom,) maps each atom to its token index.
- `token_has_frame` — (N_token,) int.
- (note: `atom_coordinate` and `atom_is_polymer` are stripped by `get_clean_full_confidence`.)

> For the M-C predictor-conditional features (ipTM/PAE/pLDDT), the parser needs BOTH files:
> `summary_confidence_*` for iptm/ptm/chain_* and `full_data_*` for `token_pair_pae` +
> `atom_plddt`. **Tasks 5–6 must add `--need_atom_confidence true`** to the campaign command
> (it was off for the smoke summaries; the fixture below includes a full_data sample).

## (C) MSA wiring (for Tasks 5–6)

- **Precomputed-MSA input field (per protein chain):** `proteinChain.unpairedMsaPath` — an
  absolute path to a ColabFold-style `.a3m`. Protenix emits a ready-to-reuse
  `<name>-update-msa.json` after its first MSA search, with `unpairedMsaPath` filled in per
  chain; feeding that file as `--input` skips the server entirely
  ("do not need to update msa result, so return itself"). (Protenix-server mode also uses a
  `pairedMsaPath`/`precomputed_msa_dir`; ColabFold mode here produced only unpaired per-chain MSAs.)
- **Disable the built-in MSA search:** there is no single "off" flag — either pass an input
  whose chains already have `unpairedMsaPath` (above), or set `--use_msa false` (drops MSA
  features entirely; not what we want).
- **MSA host (PARCC-specific):** compute nodes **cannot resolve `protenix-server.com`** (the
  default MSA host), so the default `--msa_server_mode protenix` fails with NameResolutionError.
  Use the public ColabFold server: `export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com`
  **plus** `--msa_server_mode colabfold` (the flag alone only switches result parsing; the host
  comes from the env var). dgx nodes CAN reach api.colabfold.com. MSA depth obtained: 13026.
- MSA artifacts land under `<out_dir>/3OGO__3OGO/msa/` (per-chain `non_pairing.a3m`, plus a
  `complex/pair.a3m`).

---

## Test fixture

A trimmed real output is committed at `tests/fixtures/protenix/3OGO__3OGO/`:
- `3OGO__3OGO_summary_confidence_sample_0.json` — verbatim (small).
- `3OGO__3OGO_full_data_sample_0.trimmed.json` — the three (372,372) matrices
  (`token_pair_pae`, `token_pair_pde`, `contact_probs`) downsampled to the first 24×24 tokens
  and `atom_plddt`/`atom_to_token_idx` truncated to the first 200 atoms, so the parser can be
  TDD'd against real keys/dtypes without a 6 MB blob. **Downsampling is lossy — do not use the
  trimmed PAE values as ground truth, only the schema.** The full untrimmed file lives under
  `data/raw/predictions/protenix/_smoke/3OGO_out_fulldata/` (gitignored).
- `3OGO__3OGO_sample_0.cif` — top-ranked structure (chain resolution oracle; pLDDT in B-factors).

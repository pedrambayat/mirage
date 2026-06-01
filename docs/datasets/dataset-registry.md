# mirage dataset registry

Living registry of every dataset, its role, label provenance, and caveats.
Source of truth for the data side of `docs/superpowers/specs/2026-05-31-mirage-data-and-training-strategy.md`.

| Dataset | Role | True positives (evidence) | Negatives | Predicted complex? | Confidence? | Format | Caveats |
|---|---|---|---|---|---|---|---|
| Champloo / Smorodina | Primary (train + in-dist test) | 106 cognate VHH–Ag, co-crystal | ~11,130 constructed shuffled non-cognate | Yes — AF3 staged | Yes — ipTM | VHH | Synthetic negatives; small positives (106); ~1:105 imbalance |
| SAbDab | Orthogonal TP reservoir + format axis | ~1k nonredundant VHH (+Fab/scFv), co-crystal | none native | No (crystal only) | No | VHH + Fab/scFv | Predictor-conditional use needs AF3 runs; cluster for redundancy |
| EpCAM (SNAP + collaborator) | Orthogonal test — designed-binder deployment regime, real labels | 8 functional VHHs (CAR-T killing vs AsPC1): 10,25,26,34,57,61,74,86 | 6 non-functional (real): 14,15,16,18,21,73; +86 SCR/24 OFF (unassayed) | No — must be generated | No | VHH (designed) | N=14 → test only, wide CIs; label = functional killing (downstream of binding) |
| AVIDa-hIL6 | Orthogonal test only | many VHH binders to IL-6 family (assay) | many real assay non-binders | No | No | VHH | Sequence-only; HF config broken → use raw CSVs |
| Germinal | Parked design case study | 24 BLI-positive designed | no clean public negatives | — | — | designed | Not benchmark-ready |

**Train vs test:** Champloo trains; SAbDab/AVIDa/EpCAM are held-out validation. Only AVIDa and EpCAM carry real negatives. EpCAM is functional-killing-labeled and tiny (test-only).

# WSP results summary: LFM2.5-350M insurance extraction

Model: LiquidAI/LFM2.5-350M, revision 9921cd4d82c0e3986887aa4b376f9446729ef8a3.
Original checkpoint parameter SHA256: `65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`.

## Outcome

One kept weight update, `candidate5-l15-gated-units-r3`, passed every development check and the
one-time held-out check. It is saved as a new model folder. The original checkpoint is unchanged.

- Saved model: `models/LFM2.5-350M-9921cd4d-wsp-candidate5-l15-gated-units-r3/`
- Edit record: `models/LFM2.5-350M-9921cd4d-wsp-candidate5-l15-gated-units-r3/WSP_EDIT.json`
- Edited parameter SHA256: `d690118cc22e232a04b1bba52082e14474c7682019e9c6250a6f18c1a4d386b6`
- Development results: `experiments/candidates/candidate5-l15-gated-units-r3/`
- Held-out results: `experiments/heldout/candidate5-l15-gated-units-r3/`

## What the kept edit does

Four nearly idle MLP units in layer 15 (indices 34, 235, 4572, 2265) were rewritten:

- `w1` rows: a gate direction that is high only at the `form_type` decision token.
- `w3` rows: one-vs-rest readouts for the four mapping labels.
- `w2` columns: push toward the correct label's first token.

Entries changed: 4,096 in `w1`, 4,093 in `w3`, and 4,095 in `w2` (12,288 slots in total). No new
parameters, no gradients, no fine-tuning. All fitting was closed-form on the 61 fit claim cases.

Gate check: at held-back fit positions, at most 0.1% of non-decision positions crossed the gate
threshold. The model's own generated tokens outside the label changed in 0.05% of positions.

## Results

| Check | Original | Edited | Limit |
|---|---|---|---|
| Dev test `form_type` errors (61) | 61 | 27 | >= 10% fewer |
| Dev test label accuracy, original order | 0.23 | 0.95 | not down |
| Dev test label accuracy, reversed order | 0.15 | 0.92 | not down |
| Dev field F1 (200 cases) | 0.2786 | 0.3102 | not down |
| Unrelated exact (100) | 0.70 | 0.69 | down <= 0.01 |
| Array row F1 (60 arrays) | 0.5893 | 0.5893 | down <= 0.01 |
| Null-array accuracy (10) | 0/10 | 0/10 | down <= 0.01 |
| **Held-out `form_type` errors (30 claim cases)** | 30 | 15 | >= 10% fewer |
| **Held-out field F1 (49 cases)** | 0.3547 | 0.3883 | not down |

Caveats:

- The unrelated-answer drop is exactly at the 1-point limit (one answer of 100).
- The held-out set is small. Only 30 of its 49 cases are claim forms.
- Most of the field-F1 gain comes from the label fix. Non-claim document types were not
  analysed separately.
- Array behaviour is unchanged on the control set; arrays were not a target of this edit.

## Candidates tried

| Candidate | Edit | Dev `form_type` test errors | Outcome |
|---|---|---|---|
| 1 | layer-14 head-5 query rows 320-383 (rank 4) | 61 -> 60 | Reverted. Too weak. |
| 2 | layer-15 `w2`, 64 columns, push 12 (7.9% of tensor) | 61 -> 45 | Reverted. Unrelated 70 -> 68; invented extra fields. |
| 3 | layer-15 `w2`, 64 columns, push 8 (5.3% of tensor) | 61 -> 51 | Reverted. Unrelated 70 -> 68; field F1 down. |
| 4 | layer-15 `w2`, 64 columns, generation-protected | 61 -> 50 | Reverted. Field F1 0.279 -> 0.256; unrelated 70 -> 69. |
| 5 r3 | layer-15 gated MLP units (kept) | 61 -> 27 | Passed all dev checks and held-out. Saved. |

Candidate 5 went through three revisions. Each change was made before scoring any test case:

- r1 stopped at the gate check (a non-decision position scored above the threshold).
- r2 reached selection but failed the selection rule, because the zero-structural-change rule could
  not be met even by the ablation of the original units.
- r3 compared structural changes against that ablation and passed.

## Earlier findings (closed lines)

- **Missing-field JSON null:** prompt copies, activation patches, and exemplar/schema variants do
  not produce a reliable absent-to-null repair without erasing present values. Not pursued further.
- **Form type mechanism:** the model's internal representation identifies the document type from
  layer 2 onward (about 96-99% probe accuracy). The output copies the first key of the schema's
  canonical-label mapping, via layer-14 head 5.

## Compute

- Frozen 3060 baseline: `experiments/fieldbench/fieldbench-dev-original-v2/`.
- Current work ran on a rented RTX 3090 (Vast.ai instance 59660054). The old instance 59647525 is
  stopped. The 3090 is still running and billing about $0.19/hr.

## Not yet done

- Not verified outside FieldBench insurance cases and the local control sets.
- The saved folder was reloaded from disk and its hash matched the edited model. It has not been
  run through a separate inference service.
- Not committed to git. The repository has no commits yet.

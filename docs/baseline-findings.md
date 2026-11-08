# Frozen-model baseline findings

These runs establish behavior of the original `LiquidAI/LFM2.5-350M` revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. No parameters were changed. The local LawPAK seed and FieldBench development set are separate evaluations; the latter is supplemental, not a replacement for the target insurance-submission task.

## LawPAK seed

Two cases from one insured were run on an RTX 3060 in BF16. Both responses were Markdown-fenced. After fence-only normalization, one parsed and one remained invalid because numeric values such as `1,000,000` were emitted without JSON quoting. The parseable application output still failed the JSON Schema; neither case was fully correct. The expected labels are not yet human-reviewed, so these results are exploratory only.

See the ignored raw run at `experiments/baseline/baseline-vast-rtx3060/` and its fence-only diagnostic score at `metrics.fence-stripped.json`.

## FieldBench development partition

The pinned FieldBench subset contains insurance policy declarations and claim forms, with 200 development examples and 49 held out. The development run used greedy decoding, BF16, a 768-token output cap, and the same pinned model. No generation reached the token cap. Mean inference time was 1.88 seconds per case.

The project's exact field scorer and JSON Schema validator reported:

| Measure | Development result |
| --- | ---: |
| Strict JSON validity (raw response) | 0 / 200 |
| JSON validity after removing one enclosing Markdown fence | 196 / 200 (98.0%) |
| Schema-valid after fence removal | 2 / 200 (1.0%) |
| All fields exactly correct | 0 / 200 |
| Field precision / recall / F1 | 21.8% / 39.1% / 28.0% |
| Unsupported values on gold-null fields | 191 / 262 (72.9%) |

Breakdown:

| Category and source | Cases | Schema-valid | Field F1 | Unsupported / gold-null |
| --- | ---: | ---: | ---: | ---: |
| Insurance claims, real-source forms | 10 | 10.0% | 0.0% | 70 / 80 (87.5%) |
| Insurance claims, synthetic | 112 | 0.0% | 25.2% | 97 / 151 (64.2%) |
| Insurance policies, synthetic | 78 | 1.3% | 33.7% | 24 / 31 (77.4%) |

These are project-local exact-match scores, not FieldBench's published canonical benchmark score. FieldBench is mostly clean-text input; it does not measure PDF OCR quality. The 10 real-source cases are too few for a stable real-document estimate.

## Reproducible failure examples

- `fema_proof_of_loss_086`: the gold answer is `Proof of Loss` and every other field is null. The model returned `Demand Letter`, filled claimant/employer placeholders, used a form expiration date as the loss date, and supplied a zero claimed amount.
- `fema_proof_of_loss_hpcc`: the output included empty strings, the string value `"null"`, and fields copied from form/schema language that were not in the target schema.
- `synth_binder_001`: the model got the insured, policy number, insurer, and type right, but returned dates in `MM/DD/YYYY`, money as currency strings instead of numbers, `"TBD"` instead of null, and an extra `source_document_text` property.
- `wc_froi_alabama`: the response was malformed JSON, including a key without a value separator.

All four raw generations are in the ignored `experiments/fieldbench/fieldbench-dev-original-v2/predictions.jsonl`; per-case scores and schema errors are alongside them. The 49 held-out examples were not generated or scored.

## Failure taxonomy

1. **Output contract:** all 200 outputs used Markdown fences despite a JSON-only instruction; four remained syntactically invalid after the fence was removed.
2. **Null discipline:** the model filled gold-null fields with placeholders, dates, zeros, empty strings, the string `"null"`, and unrelated field values.
3. **Schema fidelity:** most outputs had missing required properties, unsupported extra properties, wrong primitive types, or noncanonical dates.
4. **Field-role and semantic confusion:** form labels and nearby document details were assigned to fields with different meanings (for example, a form expiration date was used as a loss date).
5. **Array handling:** these FieldBench schemas do not test long output arrays. The LawPAK quote has a small coverage schedule, but its original-model output was malformed JSON, so array accuracy there is not yet a clean causal diagnosis.

This taxonomy is descriptive, not causal evidence for a particular tensor. No weight edit should be selected from it alone. Next, inspect matched correct/failed contexts and intermediate activations on the development set, then test one small intervention at a time. Do not use the held-out split to choose edits.

# form_type label mapping: frozen diagnostics

All runs use the 122 development `claim_form.yaml` cases only, BF16, greedy, parameter hash
`65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f` before and after each run.
Held-out cases were not read.

## Why this target

In the original development run `form_type` is never correct (0/122). Every claim-form title in
the first 400 characters names its canonical label or a listed synonym, so a keyword rule scores
122/122. The model writes "Demand Letter" or "Subrogation Demand" in most cases. "Demand Letter"
is the first key of the canonical-label mapping dict inside the field description.

## decision-v1 (`scripts/probe_form_type.py`, `experiments/form_type/decision-v1/`)

Prefix ```` ```json\n{\n  "form_type": "```` then the label decision token.

| arm | greedy exact label | first-token choice among 5 labels |
| --- | --- | --- |
| original mapping order | 1/122 | Demand Letter 118/122 |
| mapping order reversed | 19/122 (all are Proof of Loss gold) | Proof of Loss 122/122 |

The model copies whichever key is listed first. Sequence log-probability across options prefers
the one-token ` null` over every multi-token label, so the constrained-null column is a length
artefact and is not used.

A nearest-centroid leave-one-out probe on the residual at the decision token recovers the gold
form type at 0.96 to 0.99 from layer 2 onward (0.18 at layers 0 and 1). The information is
present; it does not reach the output.

Logit lens, gold first token minus "Dem", averaged over the 100 non-Demand cases, stays positive
through layer 13 and falls to -5.2 at layer 14. Direct attribution: layer14.attention -5.05
(largest mid-stack term).

## heads-v1 (`scripts/probe_form_type_heads.py`, `experiments/form_type/heads-v1/`)

Direct attribution inside layer 14 attention: head 5 -4.95, head 14 -1.26, head 11 -0.92.
Head 5's strongest source token is the first mapping key in 24/24 sampled cases.

Zeroing head 5 at every position (same output as zeroing its 64 `out_proj` input columns):
first-key choice 0.97 to 0.07, but gold label choice only 0.21 to 0.29, and greedy exact label
0.008 to 0.074. The model then copies other text ("VIA CERTIFIED MAIL", "DOCUMENT-LEVEL",
"Other"). Zeroing is not a candidate: under the 10% bar and likely harmful to other copying.

## pointer-v1 (`scripts/probe_form_type_pointer.py`, `experiments/form_type/pointer-v1/`)

Eager attention. Head-output reconstruction from weights and values agrees to 9e-4.

| head-5 output at decision token | original: gold chosen | reversed: gold chosen |
| --- | --- | --- |
| unmodified | 0.04 | 0.00 |
| natural first-key weight moved to gold key | 0.26 | 0.12 |
| gold-key value vector only | 0.85 | 0.97 |
| wrong-key value vector only (control) | 0.01 (wrong label 0.67) | 0.00 (wrong label 0.89) |

Greedy exact label with the gold-key value only: 0.54 original, 0.48 reversed.

Head 5 writes whichever key it reads. Its selection carries no document-type signal: the gold key
ranks first among non-first keys in 41/100 cases, near chance. The fault is head 5's query/key
selection, not the value or output path.

## Candidate 1: layer-14 head-5 query edit (`scripts/run_candidate1.py`) — failed

Tensor `model.layers.14.self_attn.q_proj.weight` rows 320..383, rank 4, closed-form ridge
least squares toward the gold mapping key, protected by other-position input statistics. Split:
122 development claim cases by sha256(id), 61 fit / 61 test. Edit in GPU memory only.

- r1 (`experiments/candidates/candidate1-l14h5-query/`): aborted at the 25% norm cap before any
  accuracy was computed. Smallest setting 0.42.
- r2 (`experiments/candidates/candidate1-l14h5-query-r2/`): only the grid changed (alpha 0.1 to
  0.25, mu 3 and 30). Every setting left fit label accuracy at 0.172. The selected edit was 4.1%
  relative Frobenius, 57,492 bf16 entries changed.

| metric | original | edited |
| --- | --- | --- |
| test form_type errors, full generation | 61/61 | 60/61 |
| development field F1 | 0.2786 | 0.2835 |
| unrelated exact | 0.70 | 0.70 |
| array row F1 | 0.589 | 0.592 |
| null arrays | 0/10 | 0/10 |

Target reduction 1.6%, below 10%. Fails. A query rotation small enough to pass the cap does not
redirect head 5. Only 40 of the 61 test cases emit the `form_type` key at all.

Original checkpoint hash restored after the run. Held-out not used. 2 of 3 candidates remain.

## Candidate 2: layer-15 w2 columns (`scripts/run_candidate2.py`) — failed

Tensor `model.layers.15.feed_forward.w2.weight`, 64 columns (65,536 entries), same columns as
the fit-only dry run. Closed-form ridge push toward the gold label token. Caps: 10% of the
whole tensor and 2.5% change at protected positions. Selected on the fit half: gain 12, protection
10. Applied change 7.93% of the tensor and 67.1% of those 64 columns (65,236 bf16 entries).
GPU memory only.

| metric | original | edited |
| --- | --- | --- |
| test form_type errors, full generation | 61/61 | 45/61 |
| test first-token label accuracy | 0.230 | 0.459 |
| test reversed-mapping label accuracy | 0.148 | 0.361 |
| development field F1 | 0.278639 | 0.278615 |
| unrelated exact | 70/100 | 68/100 |
| array row F1 | 0.589 | 0.592 |
| null arrays | 0/10 | 0/10 |

The form_type error drop is 26% relative, above the 10% bar. The edit still fails. Unrelated
accuracy fell by 2 points (`plural-18` "ships" became "groups"; `code-06` "1" became "True"),
past the 1-point limit. Field F1 also fell, by 0.000025: true positives rose 512 to 527 while
false positives rose 1829 to 1922. Of the 16 test cases repaired, several became "Loss Run";
20 test cases still omit the key.

Original checkpoint hash restored. Held-out not used. 1 of 3 candidates remains.
Results: `experiments/candidates/candidate2-l15w2-columns/`.

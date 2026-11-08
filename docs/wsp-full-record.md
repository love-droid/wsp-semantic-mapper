# WSP full record: what we did, why, and what improved

Date of this record: 8 November 2026.
Repo: `wsp-semantic-mapper`.
This note is the long account. The short table is in `docs/wsp-results-summary.md`.

## 1. The job

Improve LiquidAI/LFM2.5-350M (revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`) on insurance document extraction, using only small direct edits to existing weights.

Hard limits that stayed in force the whole time:

- No fine-tuning, LoRA, adapters, new parameters, new layers, tokenizer changes, backprop, or distillation.
- No broad weight edits. Name the tensor, the entry count, and the size of the change before applying it.
- Keep the original checkpoint. Any kept edit is a new folder.
- Do not look at the 49 held-out FieldBench cases while choosing an edit. Score them once, only after a candidate has already passed every development check.
- Do not call something an improvement without a measured before/after, a regression check, and (for a kept edit) the held-out run.
- Respect that LFM is a hybrid of short convolutions and attention, not a plain transformer.

Pass bar, agreed before any edit:

- The target error falls by at least 10% relative.
- Correct behaviour that already worked does not get worse.
- Unrelated-task accuracy, array row F1, and null-array accuracy each fall by at most 1 percentage point.
- Development field F1 does not fall.

An early cap of 3 candidate edits was agreed. After candidate 2, the instruction was to keep going until there was a real improvement, with approval to use further candidates. Five distinct edits were scored. Only the last one was kept.

## 2. The model

| Item | Value |
| --- | --- |
| Parameters | 354,483,968 |
| Layers | 16 |
| Hidden size | 1024 |
| Vocabulary | 65,536 |
| Embeddings | tied to the output head |
| MLP | `w2(silu(w1(x)) * w3(x))`, intermediate size 4,608 on the tensors we edited |
| Attention | 16 heads, head size 64, 8 key/value heads |

Layer types, zero-based: conv, conv, attention, conv, conv, attention, conv, conv, attention, conv, attention, conv, attention, conv, attention, conv. So layer 14 is attention and layer 15 is a convolution plus an MLP. Layer 15 is the last block. Its MLP output is added into the residual that the final norm and the output head read.

Inference settings for every comparison: bfloat16, greedy (no sampling), repetition penalty 1.0, left padding when batched. Batch size can change the generated string, so an edit is only compared with an original run from the same script on the same GPU. Scripts refuse to overwrite an existing output folder. Parameter hash before and after every frozen run: `65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`.

## 3. Data

**FieldBench insurance split**

- Corpus `fieldbench/corpus`, revision `7247f4204e4a60a80875d30ef06ea54f8006304a`.
- Schema commit `57cdb82efaa6c0ba856679a56cf63e614fa1959b`.
- 200 development cases, 49 held-out cases.
- Local path: `data/external/fieldbench-v0.4.2/`.
- Development SHA256 `4ef4f18c0bd62fda839ccdfc6859373b23fafd10ca8ca8e21ed477e2b2eedeee`.
- Held-out SHA256 `02cd4a05f92c65e8497d9b52d87ed0fae69ef49d9258bcdcb4bdba0e776bff40`.
- Two schemas in the development set: `claim_form.yaml` (122 cases) and `policy_declarations.yaml` (78 cases).
- Claim-form gold labels: Loss Run 51, Employer's First Report 30, Demand Letter 22, Proof of Loss 19.
- The prompt is the project prompt in `scripts/run_fieldbench.py`: one JSON object, no markdown, null when the document does not support a value, plus the full JSON schema and the source text.
- Scoring is the project scorer in `scripts/evaluate.py`, not a claim that this is the official FieldBench leaderboard number.

**How the 122 claim cases were split**

Sorted by SHA256 of the case id. First 61 are the fit half (the only cases allowed to choose units, directions, and the strength of an edit). Last 61 are the development test half. The same split was reused for every candidate so later edits could not quietly train on cases that earlier edits had treated as test.

**Controls, written before any edit** (`scripts/build_controls.py`)

- `data/controls/unrelated-v1.jsonl`, 100 items. SHA256 `8caf841e98f38e9a9fa67dd821e06e14abab09e44eb506afe4a0be3a086560f2`. Twenty each of arithmetic, regular plurals, copy-the-string, small Python values, and fenced snippets.
- `data/controls/arrays-v1.jsonl`, 70 items. SHA256 `7ccf9dec28d1d13e6bd1797a8697207c488c845717f13c227db6e957676f237f`. Practice areas, coverage terms, addresses, and 10 cases whose practice-area list is null.
- These are not FieldBench. FieldBench development scores had no array rows, so the array file is the regression check for arrays.

**What was not used to choose edits**

The 49 held-out cases were not opened for diagnosis. They were scored once, after candidate 5 had already passed development.

## 4. Where the model started

Frozen development run, 200 cases, RTX 3060, later repeated on the 3090 with the same parameter hash. Artifacts: `experiments/fieldbench/fieldbench-dev-original-v2/`.

| Measure | Value |
| --- | --- |
| Raw JSON (no fence removal) | 0% |
| JSON after removing a markdown fence | 98% |
| Schema-valid after fence removal | about 1% (2 of 200 schema-valid on a later recount; 0 fully correct) |
| Exact field F1 | 0.279 |
| Gold-null fields filled with a value | 191 of 262 (72.9%) |

Error counts on the saved development scores:

| Kind | Count | What it means |
| --- | --- | --- |
| unexpected_value | 1,207 | a field the gold object does not have |
| value_mismatch | 473 | right field, wrong value |
| missing_value | 339 | gold has a value, prediction does not |
| unsupported_value | 191 | gold is null, prediction invented a value |

Schema errors: 406 missing required keys, 206 bad dates (mostly `MM/DD/YYYY` instead of `YYYY-MM-DD`), 82 extra properties, 22 type errors, 16 enum errors.

`form_type` on the 122 claim forms: 0 correct in that saved run. The document title in the first 400 characters already names the right label or a listed synonym, so a keyword over the title would score 122/122. The model was not failing to see the title. It was ignoring it.

Control baseline on the 3090, same hash:

| Check | Value |
| --- | --- |
| Unrelated exact | 70 of 100 (50 of 80 exact items, 20 of 20 fences) |
| Array row F1 | 0.589 (96 expected rows, 128 predicted, 60 matched) |
| Exact array row sets | 0.133 |
| Null arrays correct | 0 of 10 |

## 5. Compute

- Old machine: Vast.ai instance 59647525, RTX 3060. It could not be restarted because the GPU stayed in use. Its `/workspace` was copied onto the new machine at `/workspace/from-old-59647525` (model, FieldBench data, earlier experiment outputs). The instance was then stopped. It was not destroyed.
- New machine: Vast.ai instance 59660054, RTX 3090, about $0.185/hr, image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, PyTorch 2.7.1+cu128, Transformers 5.22.2. Still running when this note was written.
- The causal-conv1d fast kernel was not installed. The PyTorch reference implementation was used and was not swapped mid-comparison.

## 6. Closed line: JSON null for missing fields

This was the first target because the handoff note named missing-field hallucination. It was closed without a weight edit.

What was tested, all with frozen weights, on 24 synthetic present/absent pairs:

- Activation swaps at layer 11 MLP, layer 13 MLP, layer 10 attention, and the final residual as a control that must reproduce logits exactly. Self-patch difference was 0.
- Donor documents that say "not provided", "unknown", or that simply omit the fact. Free generation never emitted a JSON null. The model copied the words "not provided" or "unknown", or wrote a quoted `"null"`.
- The token race at the decision `{"field":`. The quote token (id 997, text ` "`) always beat the null token (id 3996, text ` null`). Patching an absence activation into a present case pushed the present value toward null harder than it pushed the absent case. Layer 10 flipped a few tokens; it did not repair absent fields while keeping present values.
- Prompt variants: a schema note that null is allowed, one null exemplar, two exemplars. Free generation still wrote 0 JSON nulls on absent fields. Forcing the prefix after one null exemplar nulled the absent cases and also 19 of 24 present values. Transferring those activations back onto the original prompt did not specifically repair absence (layer 11 flipped 0; layer 10 flipped 1 and tied the present value).

Conclusion, written down before any weight change: this behaviour is prompt copying, not a small circuit that can be nudged. Editing weights for it would erase real values. Artifacts: `experiments/null_pairs/` and `docs/null-*-findings.md`.

## 7. Why `form_type` was the next target

After nulls were closed, the development errors were ranked by how systematic they were. A systematic error is the kind a few weights might fix.

`form_type` was the clearest:

- 0 of 122 correct, while the title states the answer.
- The schema text lists canonical labels in a fixed order. "Demand Letter" is first. The model writes "Demand Letter" or "Subrogation Demand" on most claim forms.
- Reversing only the order of that mapping dict, and changing nothing else, makes the model write "Proof of Loss" on all 122 forms. So the output is copying the first listed key, not reading the document.
- Dates and missing keys are numerous (206 and 406) but were not shown to be one circuit. They were left for later on purpose.

## 8. How the edit site was narrowed

All of this is frozen. No weights changed. Only the 122 development claim forms. Scripts and folders are named in parentheses.

**The decision token** (`scripts/probe_form_type.py`, `experiments/form_type/decision-v1/`)

The model almost always starts with a markdown fence, so the label decision is the token after:

```` ```json\n{\n  "form_type": " ````

First tokens of the five labels do not collide: Proof, Em, L, Dem, Other.

| Mapping order | Greedy exact label | First-token choice among the five labels |
| --- | --- | --- |
| Original (Demand Letter first) | 1 of 122 | Demand Letter on 118 of 122 |
| Reversed (Proof of Loss first) | 19 of 122, all of them Proof of Loss gold | Proof of Loss on 122 of 122 |

A nearest-centroid probe on the residual at that token recovers the true form type at 18% in layers 0 and 1, then 96% to 99% from layer 2 through layer 15. The document type is already in the residual. It is thrown away at the output.

A logit lens (read the residual with the output head after each layer) keeps the true label ahead of "Dem" through layer 13 on the 100 non-Demand cases, then drops by about 5 logits at layer 14. Direct attribution of that drop: layer 14 attention, about −5.05. That is why layer 14 was inspected next, not an early layer and not the embedding.

**Which head** (`scripts/probe_form_type_heads.py`, `experiments/form_type/heads-v1/`)

Inside layer 14 attention, head 5 accounts for −4.95 of the push from the true label toward the first listed key. Heads 14 and 11 are smaller (−1.26 and −0.92). On 24 sampled cases, head 5's strongest source token is the first mapping key every time ("Dem" or "Proof" depending on list order).

Zeroing head 5 stops the copy (first-key choice 97% down to 7%) but the true label only rises to 29%, and greedy exact labels go from 0.8% to 7%. The model then copies other schema text ("VIA CERTIFIED MAIL", "DOCUMENT-LEVEL"). Zeroing was rejected: below the 10% bar, and it breaks copying the model needs elsewhere.

**Whether the head can be retargeted** (`scripts/probe_form_type_pointer.py`, `experiments/form_type/pointer-v1/`)

Head 5's output was reconstructed from its attention weights and value vectors (max error 0.0009).

| What head 5 is forced to read | True label chosen |
| --- | --- |
| Whatever it reads on its own | 4% (original order) |
| Its existing attention weight moved from the first key onto the true key | 26% |
| Only the true key's value vector | 85% to 97% |
| Only a wrong key's value vector | that wrong label wins (67% to 89%) |

So the write path works: the head writes the key it reads. The read path does not know the document. Among keys that are not first, the true key ranks first only 41 of 100 times, close to chance. That is why candidate 1 tried to change the query (what the head looks at) rather than the value or the output projection.

## 9. Why each candidate edited those parameters and not others

The rule was: edit the smallest site that the measurements say is causal, and only after a smaller or more obvious site has failed.

### Candidate 1 — layer 14, head 5, query rows 320–383

**Why these rows.** Head size is 64. Head 5's query lives in rows 320 through 383 of `model.layers.14.self_attn.q_proj.weight`. That is 64 × 1024 = 65,536 entries, about 0.018% of the model. The query is what chooses the source token. The pointer test said the source choice was the fault.

**How it was built.** Closed-form ridge least squares. No gradients. Fit half only. The query at the label token was steered toward the true key. Other token positions in the prompts were penalised so the query would barely move there. A 25% cap on the size of the change relative to those rows.

**What happened.**

- First grid (step sizes 0.5 to 2) was entirely above the 25% cap. Smallest change was 42% of those rows. Aborted before any accuracy was scored. Not counted as a used evaluation.
- Smaller grid (steps 0.1 to 0.25). Every setting left fit-half label accuracy at 17%, the same as doing nothing useful. The smallest legal change (4.1% of those rows, 57,492 entries actually different in bfloat16) was applied and scored.
- Development test `form_type` errors: 61 to 60. Field F1 0.2786 to 0.2835. Unrelated 0.70 to 0.70. Arrays unchanged-to-slightly-up. Failed the 10% bar.
- Weights restored. Hash matched the original.

**Why we left this site.** A query change small enough to be local does not move head 5 onto the right key. The pointer test already showed that merely moving its existing attention weight is a weak effect (26%). Making the query point at the right place needs a large rotation, and the cap forbids that. Also, 20 of 61 test cases never emit the `form_type` key at all, so a change that only acts at that key cannot help them unless the rest of the generation still reaches the key.

### Candidates 2 and 3 — layer 15 MLP down-projection, 64 columns

**Why this tensor.** The residual already contains the document type from layer 2. The last place that residual is added to, before the output head, is layer 15's MLP. `w2` is the down-projection: shape 1024 × 4608. Editing columns of `w2` adds a vector to the residual. If a column's input unit is active at the label token and quiet elsewhere, that column can push one label without rewriting the rest of the model.

**Why 64 columns, and which ones.** On the fit half only, units were ranked by how well they separate the four gold labels at the decision token, divided by how active they are at other prompt positions. The top 64 were kept. 64 × 1024 = 65,536 entries, the same budget as candidate 1, but at the output end where the class signal is already formed.

**Why the push size.** A grid of logit pushes (4, 8, 12) and protection strengths was scored on the fit half only. Larger push, higher fit accuracy, larger side effect.

| Push | Protection | Fit label accuracy | Change vs whole `w2` | Change at other positions |
| --- | --- | --- | --- | --- |
| 4 | 10 | 25% | 2.6% | 0.8% |
| 8 | 10 | 39% | 5.3% | 1.5% |
| 12 | 10 | 57% | 7.9% | 2.3% |
| 12 | 1 | 87% | 18% | 5.4% |

Candidate 2 used push 12, protection 10 (the strongest setting inside a 10%-of-tensor cap and a 2.5% cap on other-position outputs). Candidate 3 used the next smaller one, push 8, protection 10, with no new search.

**What happened.**

| | Original | Candidate 2 | Candidate 3 |
| --- | --- | --- | --- |
| Test `form_type` errors | 61 | 45 | 51 |
| Cases fixed / broken | | 16 / 0 | 10 / 0 |
| Field F1 | 0.2786 | 0.2786 | 0.2752 |
| Unrelated | 70/100 | 68/100 | 68/100 |
| Outputs changed out of 200 | | 102 | 85 |

Both cleared the 10% error-reduction bar and both were reverted. The same two unrelated items failed both times: plural "ships" became "groups", and a code answer "1" became "True". Field F1 failed because extra invented fields rose faster than correct fields. One case, `structural_loss_run_gap4_001`, gained 41 false fields under both edits. The edit was not gated: those 64 units are active while the model is writing, so the label push leaked into other documents, including policy declarations that have no `form_type`.

**Why we did not keep shrinking this edit.** Candidate 3 was already the smaller push and still broke the same unrelated answers. Another notch down would likely miss the 10% bar (candidate 1 had already shown that a weak output nudge does not survive full generation). The failure was the lack of a switch, not the choice of layer.

### Candidate 4 — same kind of columns, but chosen to be quiet while generating

**Why.** Candidates 2 and 3 changed about half of all development outputs. The hypothesis was that the units were chosen using prompt positions only, so they still fired during generation. Candidate 4 added the model's own generated tokens (fit-half claim outputs, plus 40 generic prompts that are not in the control suite) to the "stay quiet" measurement, and rejected any setting that changed the greedy token at more than 1% of those positions.

**What happened.** The fit-half flip rate looked excellent (0.40% at the chosen push of 16). The full run did not. Test errors 61 to 50 (an 18% reduction, above the bar). Field F1 fell from 0.279 to 0.256. True positives fell (512 to 498) and false positives rose (1,829 to 2,059). Unrelated fell 70 to 69. 110 of 200 outputs changed, including 24 policy documents. Example: a loss-run case that had been emitting the claim schema switched to inventing `"source_document_txt"` and a claims table.

**Why this site was abandoned.** Protecting the columns statistically is not the same as a switch. A column that is quiet on average can still be large on the tokens that decide the shape of the JSON. The next measurement asked whether a real switch exists.

### The gate measurement (`scripts/probe_gate.py`, `experiments/form_type/gate-v1/`)

Question: is there a direction in the MLP input, at layer 14 or 15, that is high at the label decision and low at every other position we care about?

Method: ridge direction trained on one half of the fit cases' positions, scored on the other half. Positions included prompt tokens, the model's own claim-form outputs (excluding the form_type line itself), and the 40 generic prompts.

Result, both layers, both directions of the split: the lowest decision score sat 8 to 11 standard deviations above the highest protected score. Zero protected positions on the held-back half reached the decision range. The same residual also reads out the gold label at 98% (layer 14) and 98% (layer 15) with a nearest-centroid probe.

That is the reason the kept edit is a gate plus a readout plus a push, and the reason it sits in layer 15: the last MLP can add into the residual the output head reads, and its input already separates "this is the label token" from "this is any other token" cleanly enough to use as a switch.

**Why four units, and why these indices.** An MLP unit here is one row of `w1`, one row of `w3`, and one column of `w2`. There is no bias on these linears, so a new behaviour has to be written into existing units. Units were ranked by how little they contribute on protected positions (mean absolute activation times the norm of their `w2` column). The four quietest in layer 15 were 34, 235, 4572, and 2265. Four units because there are four canonical labels. Their typical contribution is about half the median unit, and their peak activation on protected positions is small (about 0.07 to 0.19). They were the least costly place to overwrite. We did not add units. We did not touch attention, because candidate 1 had already shown the attention read is the wrong knob, and because the class signal is already in the residual by layer 15.

### Candidate 5 — the kept edit

Tensors, and only these:

- `model.layers.15.feed_forward.w1.weight` rows 34, 235, 4572, 2265. The gate. Same direction on all four rows.
- `model.layers.15.feed_forward.w3.weight` rows of those units. One-versus-rest readout, one label each: Demand Letter, Employer's First Report, Loss Run, Proof of Loss.
- `model.layers.15.feed_forward.w2.weight` columns of those units. The push from that label's first token, scaled by the gain.

Entry budget: 3 × 4 × 1024 = 12,288. After bfloat16 rounding, 4,096 + 4,093 + 4,095 entries differ. No other parameter was writable in the script; an assertion fails if anything outside those rows and columns changes.

How each piece was fit, fit half only, closed form, no gradients:

1. **Gate.** Least squares on the MLP input: +1 at decision tokens, −1 at protected positions, with the two groups balanced so the thousands of protected positions do not drown the decisions. Positions that landed near the boundary were up-weighted by 10, five rounds. The direction was then scaled so the weakest decision scores +6. Held-back check: at most 0.1% of held-back protected positions may score above −2. On the full fit set the protected max landed at −3.34 and the decision minimum at +6.00.
2. **Readout.** Each `w3` row predicts 1 on decisions of its label and 0 on other decisions and on protected positions. Fit-half readout accuracy was 100%.
3. **Push.** `w2` column adds a fixed number of logits toward that label's first token and away from the other labels, divided by the average gate activation (6.56) so the gain is in logit units at the decision.

**Why gain 16.** Grid on the fit half, scored before any test case was touched:

| Gain | Fit label accuracy | Generated tokens changed | Early-token changes |
| --- | --- | --- | --- |
| 0 (delete the four units, write nothing) | 18% | 0.040% | 2 |
| 8 | 79% | 0.040% | 1 |
| 12 | 97% | 0.040% | 2 |
| 16 | 100% | 0.048% | 2 |
| 24 | 100% | 0.032% | 2 |

Selection rule: highest fit accuracy among gains that change at most 0.1% of protected generated tokens, and that cause no more early-token changes than the gain-0 ablation. Ties take the smaller gain. Gain 16 is the smallest gain that scores 100% on the fit half inside that rule. Gain 24 does not improve the fit score and was not chosen. Gain 0 is the control that the four units were not already doing this job; it scores the same 18% as the untouched model.

**Two revisions that happened before the test half was scored**

- r1 stopped at the gate check. One held-back half had a protected score of +0.07 against a decision minimum of +0.85, so the switch was not clean. No accuracy was computed.
- r2 cleaned the gate (reweighting) but required zero early-token changes. The gain-0 ablation itself changes 2 early tokens, because those four units, idle as they are, are not exactly zero. Every real gain also sat at 1 or 2. The rule could not be passed by any edit of these units. No test evaluation.
- r3 compares early-token changes with the ablation instead of with zero. That is the run that was scored.

## 10. Evaluations of the kept edit

Same script, same GPU, same batching, original then edited, weights restored in memory afterwards and the edit reloaded from the saved tensor for the held-out run. Held-out was not loaded until `checks.all_pass` was true on development.

**Development (200 FieldBench cases, of which 61 claim forms are the test half)**

| Check | Original | Edited | Bar | Result |
| --- | --- | --- | --- | --- |
| Test `form_type` errors | 61 of 61 | 27 of 61 | at least 10% fewer | 56% fewer |
| Test label accuracy, normal schema order | 23% | 95% | not down | pass |
| Test label accuracy, reversed mapping | 15% | 92% | not down | pass |
| Field F1, all 200 cases | 0.2786 | 0.3102 | not down | +11% relative |
| Unrelated exact | 70/100 | 69/100 | drop at most 1 point | exactly at the limit |
| Array row F1 | 0.5893 | 0.5893 | drop at most 1 point | unchanged |
| Null-array accuracy | 0/10 | 0/10 | drop at most 1 point | unchanged |

**Held-out, scored once (49 cases, 30 of them claim forms)**

| Check | Original | Edited |
| --- | --- | --- |
| `form_type` errors | 30 of 30 | 15 of 30 |
| Field F1 | 0.3547 | 0.3883 |
| True positives / false positives / false negatives | 158 / 388 / 187 | 173 / 373 / 172 |
| Schema-valid rate | 0.041 | 0.041 |

Held-out claim outputs before the edit were Demand Letter 12, missing key 11, Subrogation Demand 5, unparseable 2. After: missing key still 11, Employer's First Report 8, Demand Letter 5, Loss Run 2, the long synonym "Loss Run / Claims History Report" 2, unparseable 2. The 11 cases that never emit the key are untouched, which is expected: the gate only fires when the model is about to write the label.

The saved file was reloaded from disk and its parameter hash matched the edited model (`d690118c…`). The original folder was not written.

Saved at:

- `models/LFM2.5-350M-9921cd4d-wsp-candidate5-l15-gated-units-r3/`
- edit card: `WSP_EDIT.json` in that folder
- development run: `experiments/candidates/candidate5-l15-gated-units-r3/`
- held-out run: `experiments/heldout/candidate5-l15-gated-units-r3/`

## 11. Improvement summary

Percentages a reader can use without the tables:

- **Form type on the development test forms:** from 0 right out of 61 to 34 right out of 61. Errors down 56%.
- **Form type on held-out claim forms:** from 0 right out of 30 to 15 right out of 30. Errors down 50%.
- **All-field score on 200 development documents:** 27.9 to 31.0 out of 100, about 11% higher.
- **All-field score on 49 held-out documents:** 35.5 to 38.8 out of 100, about 9.5% higher. Fifteen more fields correct, fifteen fewer invented fields, fifteen fewer missed fields.
- **Unrelated questions:** one answer worse out of 100 (70 to 69). This sits on the agreed limit, not under it.
- **Arrays:** no change.
- **Schema validity:** no meaningful change. The model still wraps JSON in a markdown fence and still misses required keys and date formats. This edit was not aimed at those.

What the edit is, in one sentence: four quiet units in the last MLP were turned into a switch that is on only while the model is choosing the form-type label, and that switch adds the label the document type in the residual already supports.

What it is not: a general extraction improvement. Dates, missing required keys, and invented fields are still the bulk of the errors. The held-out set is 49 documents. The unrelated drop of one item may be noise, and it may not be.

Why these parameters and not the others, in one paragraph: the document type is already represented by layer 2, so early layers were the wrong place to write it again. Layer 14 head 5 is what copies the first schema key, but its query could not be turned toward the right key with a small edit, and turning the head off made the model copy other text. Writing a label push into active MLP columns worked on the label and leaked into the rest of generation, even after those columns were chosen to look quiet on average. The MLP input at layer 15 is the first place we measured a switch that is on at the label token and off everywhere else we checked, and the four quietest units there were the ones overwritten so the rest of the layer would keep doing what it already does.

## 12. What is not done

- No check outside FieldBench insurance cases and the two local control files.
- The saved folder was hash-checked after reload. It has not been served through a separate inference script end to end.
- Nothing is committed to git. The repository still has no commits.
- The 3090 (instance 59660054) was still billing at about $0.19/hr when this note was written. The old 3060 instance is stopped, not destroyed.
- The agreed 3-candidate budget was exceeded after an explicit instruction to continue until an edit improved the model. That makes the held-out result weaker than a clean three-candidate study, because more attempts were made before the locked test was opened.

## 13. Sensible next steps

1. Run the normal FieldBench script on the saved folder, as a process that only loads that folder, and confirm the held-out numbers repeat.
2. Stop the 3090 unless another edit is about to start.
3. If more edits are wanted, the remaining systematic errors are missing required keys (406), date format (206, mostly `MM/DD/YYYY`), and invented fields (1,207). Each needs its own site, found the same way: frozen measurement first, then one named tensor, then the same bar. The form-type units should be left alone unless a new measurement says they are the cause.

# Matched null-field activation experiment

Date: 8 November 2026. Model: `LiquidAI/LFM2.5-350M`, revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. All weights remained frozen. No training, gradient computation, parameter updates, or held-out access occurred.

## Design

Created 24 controlled synthetic development pairs, eight each for insured name, current policy number, and loss date. Within a pair, the absent version removes exactly the line containing the target fact. Record ID, insurer, and distractors remain unchanged. Four field-label variants and two line orderings provide limited variation. These are constructed probes, not independent real-world evaluation cases.

For example, removing `Policy number: POL-7400` leaves a quote reference, a previous policy number, the record ID, and insurer. The correct current policy number becomes null; the record ID and insurer must remain correct.

The schema requires the target plus record ID and insurer. Full generations use the same generic extraction prompt builder as the FieldBench baseline, greedy BF16 inference, batches of 16, and a 256-token generation limit. This experiment does not compare its rates directly with the earlier 200-case benchmark.

## Full-generation failures

Scores below allow removal of one enclosing Markdown fence. The target must be present at the correct top-level key and exactly equal the expected value. Both retained fields must exactly match to count as preserved.

| Target | Present value correct | Absent value correctly null | Both retained fields correct, present / absent |
| --- | ---: | ---: | ---: |
| Insured name | 4/8 | 0/8 | 4/8 / 2/8 |
| Current policy number | 7/8 | 0/8 | 8/8 / 8/8 |
| Loss date | 8/8 | 0/8 | 8/8 / 5/8 |
| Total | 19/24 | 0/24 | 20/24 / 15/24 |

Not all absent-case failures are invented values: seven omit or misnest the target key. The remaining seventeen supply a non-null value, including two empty strings. Some generations copy schema structure instead of returning an instance.

Examples: `insured_name-0-absent` produces empty strings for all three fields. `policy_number-0-absent` uses `REC-policy_number-0` as the policy number while retaining the other fields correctly. `loss_date-0-absent` emits schema-like `required` and `properties` structures.

## Activation intervention

To isolate a value decision, append an identical assistant prefix such as `{"policy_number":` to both members of a pair. This is a diagnostic teacher-forced context, not the full free-generation behavior and not an inference solution.

Measure log probability mass of tokens beginning a JSON null literal versus tokens beginning a quoted string, including leading whitespace and merged spellings. The metric is a first-token family comparison, not the probability of a complete correct value. Partial spellings of `null` and other output types are outside the comparison; exact token-family IDs and the top five baseline tokens are saved.

Capture the final-position output of all 16 feed-forward blocks and all 16 mixing blocks (convolution or attention as implemented by LFM). At each site independently, replace this vector with the corresponding vector from the other member of the pair, then recompute the downstream output. Only the final position is changed. No weights are modified and no gradients are used.

Layer 11 (zero-based, the twelfth layer) has the strongest feed-forward effect by the sum of the absolute mean bidirectional margin shifts among the 32 tested component sites:

| Swap at layer 11 feed-forward output | Mean change in null-versus-string log-mass margin |
| --- | ---: |
| Absent-document activation into present document | +0.503 |
| Present-document activation into absent document | −0.393 |

The baseline margin favors strings in all 48 prefixed contexts. Absent-case margins range from −2.628 to −0.924. The layer-11 swaps do not flip any case to favor null. Therefore this is evidence that the site carries some presence/absence information, **not evidence of a repair**. A downstream component showing an effect is not necessarily where the failure originates.

The ranking varies by target: layer 15 feed-forward is strongest by mean absolute per-case effect for insured name, while layer 11 feed-forward is strongest for policy number and loss date. Do not assume one universal missing-field circuit from this small sample.

## Controls and integrity

- Replacing the strongest site's activation with its own saved activation reproduces every baseline logit exactly: maximum difference 0.
- Replacing the final residual vector with the paired donor vector reproduces the donor's decision margin exactly: maximum difference 0. This is a transfer sanity check, not a discovery of the failure location.
- Before/after SHA256 over named model parameter bytes is identical: `65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`.
- RTX 3060, BF16, PyTorch 2.7.1+cu128, Transformers 5.22.2; batch size 16. The measured diagnostic phase took 22.38 seconds, excluding model loading and initial setup.
- All 24 pair constructions passed checks that the target value is removed and the other expected values are preserved.

An earlier v1 diagnostic used standalone null/quote tokens and a different prefix boundary. That scoring method missed tokenization alternatives and is superseded by v2; its site ranking must not be used to choose an edit.

## Candidate and decision

The associated tensor worth further inspection is `model.layers.11.feed_forward.w2.weight`, shape `(1024, 4608)`. The installed implementation applies this projection after its gated feed-forward intermediate activations. The experiment intervened on its output vector, not on this tensor or any individual neuron.

No numeric weight patch is justified yet. Editing the whole tensor would touch 4,718,592 weights, so a full dense update is not a small intervention. Before proposing a patch, isolate a sparse intermediate feature or a narrowly targeted direction with stronger causal evidence, then check whether its effect distinguishes absent fields from valid present fields. In particular, swapping two failed null contexts cannot demonstrate recovery from a successful null representation; we still need matched successful-null donors.

Any subsequent candidate should be tested with an identical batched original-model baseline, target accuracy and required-key checks, retained-value preservation, unrelated extraction and ordinary language/code controls, and undo/sham ablations. The original 49 held-out cases must remain unused for choosing the edit. No before/after weight improvement is claimed here.

## Reproduction and artifacts

Run `python scripts/probe_null_pairs.py` in the remote repository with the existing pinned checkpoint. The script refuses to overwrite its run directory. Raw cases, full generations, decision scores, per-site swap results, activations, and integrity summary are in the Git-ignored directory `experiments/null_pairs/activation-probe-v2/`, copied back to the local project. Source is `scripts/probe_null_pairs.py`.

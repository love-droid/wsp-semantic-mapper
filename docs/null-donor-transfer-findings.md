# Successful-null donor search

Date: 8 November 2026. Model: `LiquidAI/LFM2.5-350M`, revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. All weights remained frozen. No training, gradient computation, parameter updates, or held-out access occurred.

## Result

None of the 120 predeclared donor documents produced a JSON null at the target field. The model copied absence wording as an ordinary string (`"not provided"`, `"unknown"`, `"null"`), copied a distractor or the record ID, or invented a value. Because the donor set was empty, there was no activation to transfer into the failed absent cases or into the present-value cases. No weight patch is justified.

The same run reproduced all 48 original probe generations from `activation-probe-v2` exactly, and the parameter digest matched the previous experiments:

`65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`

## Why this donor set was required

The earlier pair swap moved null-versus-string margins a little, but both the present and the absent document preferred a quoted string. Swapping two failed decisions cannot show that a successful null representation repairs a failure. This run looked for documents that keep the same schema, prompt, record ID, and insurer, and that actually make the frozen model emit null.

## Donor documents

Each of the 24 absent pairs was rewritten in five fixed ways. The extraction prompt and schema were unchanged. Selection rules were written to `selection_rules.json` before generation.

| Variant | Source change | Role |
| --- | --- | --- |
| `explicit_not_provided` | Same label, value replaced with `not provided` | Candidate |
| `explicit_unknown` | Same label, value replaced with `unknown` | Candidate |
| `natural_absence` | Fact line replaced with one sentence saying that fact is not stated | Candidate |
| `no_distractors` | Absent text with quote, previous-policy, broker, contact, and extra-date lines removed | Candidate |
| `literal_null` | Same label, value replaced with the word `null` | Copying control |

A donor would have counted only if, after at most one Markdown fence, the target key was JSON null and `record_id` and `insurer` were exact. An empty string, a missing key, and the quoted string `"null"` do not count. Schema validity was recorded and was also required for the stricter primary set. The primary set additionally required the greedy token under the shared prefix `{"<field>":` to be in the null token family.

## Free generation

Scores allow one enclosing Markdown fence. Batches of 16, greedy BF16, 256 new tokens. Original cases were generated in their own batches, in the same order as v2, and matched that run 48/48.

| Document | Target correct | Target is JSON null | Both retained fields correct | Schema valid | Target key missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Present original | 19/24 | 0/24 | 20/24 | 20/24 | 4/24 |
| Absent original | 0/24 | 0/24 | 15/24 | 17/24 | 7/24 |
| Explicit "not provided" | 0/24 | 0/24 | 18/24 | 20/24 | 4/24 |
| Explicit "unknown" | 0/24 | 0/24 | 19/24 | 20/24 | 4/24 |
| Natural absence sentence | 0/24 | 0/24 | 16/24 | 21/24 | 3/24 |
| Distractors removed | 0/24 | 0/24 | 21/24 | 21/24 | 3/24 |
| Literal `null` in the source | 0/24 | 0/24 | 17/24 | 20/24 | 4/24 |

What the model wrote instead, when the target key was present:

- "not provided" was copied as a string in 15/24 cases. "unknown" was copied as a string in 18/24.
- The word `null` in the document was copied as the string `"null"` in 13/24 cases, never as a JSON null.
- With distractors removed, all 8 policy-number cases used the record ID. All 8 loss-date cases used `2023-01-15`, a date that is not in the document. Four insured-name cases copied the schema description (`Extract only the explicitly supported value...`).
- The natural-absence sentence did not stop loss-date cases from copying the form-expiration distractor (`2027-09-10` and nearby dates).

Schema-valid outputs are common here because a wrong string still satisfies `["string", "null"]`. Schema validity is not evidence of a correct null.

## Decision prefix

The same assistant prefix as v2, `{"<field>":`, was applied to every document. The null-versus-quote score is log probability mass over the v2 token families, not the probability of a finished value. The greedy top token and a continuation from that token were saved as well.

Every one of the 168 prefixed contexts, including all 120 donors, had a greedy first token of ` "` (a quote). None had a positive null-minus-quote margin. The least negative donor margin was −0.205, on `insured_name-7-donor-literal_null`, and that continuation was still a quoted string. Absent-original margins stayed in the previous range, −2.628 to −0.924.

Under this prefix the retained fields were usually correct, including 24/24 for several donor variants. Forcing the JSON to start at the target key makes the shape easier and does not make the value null. Present values were correct in 21/24 prefix continuations.

## Transfer

Matched, mismatched, and reverse activation transfers were implemented for layer 11 feed-forward, layer 13 feed-forward, and layer 10 attention. They run only for free-generation successes. The success list was empty, so `transfers.jsonl` is empty. There is no measurement of whether a donor activation repairs an absent field or deletes a present value, because no donor activation was a successful null decision.

Controls that do not need a successful donor still ran:

- Replacing layer 11's feed-forward output with the same vector reproduced every baseline logit. Maximum difference 0.
- Replacing the final-layer residual with another example's residual reproduced that example's next-token logits. Maximum difference 0.
- Parameter digest before and after matched. Modified parameters: 0.

The measured portion of the run, after model load and the parameter hash, took 28.5 seconds on the RTX 3060. Batch size 16, BF16, PyTorch 2.7.1+cu128, Transformers 5.22.2.

## What this does not justify

No edit to `model.layers.11.feed_forward.w2.weight`, or to any other tensor, follows from this run. The layer-11 site from v2 still only has evidence from swaps between two failing contexts.

The 49 held-out cases were not read. These probes are not an unrelated-task regression test. The three JSON nulls in the existing 200-case development baseline are not clean donors either: one sits inside a schema-like `properties` object, and the other two null a gold-absent amount inside outputs that remain schema-invalid and wrong on most other fields.

A later donor has to be a frozen-model generation that already emits JSON null while keeping the other required fields. Rewording "missing" inside the document did not produce that behavior. The model treats those words as values to copy.

## Reproduction

Run `python scripts/probe_null_donors.py` in the remote repository with the pinned checkpoint. The script refuses to overwrite `experiments/null_pairs/donor-transfer-v1/`. That directory contains the selection rules, all 168 source documents, every generation, the prefix decisions and continuations, the empty transfer file, the activation tensors, and `summary.json`.

# Is a correct null reachable at all?

Date: 8 November 2026. Model: `LiquidAI/LFM2.5-350M`, revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. Weights stayed frozen. The 49 held-out cases were not read.

This was the last planned diagnostic. Earlier runs showed that absence wording is copied as a string, and that patching those activations pushes real values toward null harder than missing ones. The open question was whether any fixed prompt change makes a missing field JSON null while the paired real value survives. Three changes were written down before scoring, plus a fresh original-prompt arm.

| Arm | Change |
| --- | --- |
| `original` | The FieldBench prompt, unchanged |
| `schema_says_json_null` | One sentence appended to the existing field description: use JSON null, not a quoted string or an empty string |
| `null_exemplar_only` | One fixed example, different entity, whose target is JSON null |
| `two_exemplars` | That example plus one fixed example whose target is a real string |

Examples use Northwind Assurance, Harbor Glue Co, `EX-1001`, and `2019-01-02`. They do not use Atlas Mutual or any test value. The same examples are shown for present and absent test cases.

A pair succeeds only when the absent target is JSON null, the present target is the exact source string, and both keep record ID and insurer and validate. Free generation was required first. The decision prefix `{"<field>":` was the fallback.

## Free generation

No arm produced a JSON null. Counts are out of 24 absent cases and 24 present cases.

| Arm | Absent is JSON null | Present value correct |
| --- | ---: | ---: |
| Original | 0 | 19 |
| Schema sentence | 0 | 22 |
| Null example only | 0 | 18 |
| Two examples | 0 | 0 |

The schema sentence turned many misses into empty strings. It did not produce null. Two examples made every free generation unparseable as the target object, with all three required keys missing. Showing the model a finished null is not enough for it to emit one on its own.

## Forced value position

With the answer forced to start at `{"<field>":`, the null-example prompt emits JSON null for all 24 absent cases, with record ID and insurer correct. It also turns 19 of 24 present values into null. The five that survive are `policy_number-0` through `policy_number-4`. Their present-side margins over the quote token are small, down to 0.125 logits.

Two examples do not do this. Under the same prefix the absent policy number becomes the string `"NULL"`, and the record ID is replaced with `"NULL"`. The schema sentence still opens every value with a quote.

## Transfer back to the original prompt

The five prefix pairs that kept the present value were the only donors. Their absent-decision activations were patched into the original-prompt pairs. Layer 11 feed-forward flipped none of them. It moved the original present cases toward null by more than the absent cases.

Layer 10 attention flipped one absent case, `policy_number-4`. The paired present case landed in a tie: null logit minus quote logit went from −4.875 to 0.000, so the quote token won only by a tie-break. The other four present cases were left within 0.625 logits of null.

Self-patch and final-residual checks matched exactly. Parameter digest before and after:

`65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`

Measured time after the hash was 35.9 seconds.

## Decision

Do not update weights. A correct null shows up only when the prompt already contains a null example and the value position is forced open, and in that setting most real values are erased. Putting that activation back into the original prompt does not separate missing fields from real values. There is no sparse edit that these measurements identify.

No further prompt variants were added after seeing this. Reproduction is `python scripts/probe_exemplar_reachability.py`. Outputs are in `experiments/null_pairs/exemplar-reachability-v1/`.

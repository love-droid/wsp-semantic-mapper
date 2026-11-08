# Quote-versus-null token race

Date: 8 November 2026. Model: `LiquidAI/LFM2.5-350M`, revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. Weights stayed frozen. No held-out cases were read.

## Question

`donor-transfer-v1` found no finished JSON null, so there was nothing to copy. The decision prefix still has a two-token race: ` "` (token 997) beats ` null` (token 3996) on every document. This run patched each absence-document activation into the paired absent case and the paired present case, and asked whether token 3996 can win only on the absent case.

A flip counts only when the greedy token changes to 3996. That is the first token of the value, not a finished JSON object. Donors that never emitted null are included on purpose. `literal_null` remains a copying control.

## Result

Layer 11 feed-forward, the site from the earlier pair swap, flipped nothing. It moved present cases toward null by more than it moved absent cases, for every donor variant. Mean change in the null-minus-quote logit gap at layer 11, explicit "not provided":

| Destination | Mean gap change |
| --- | ---: |
| Absent | +0.229 |
| Present | +0.755 |

Layer 10 attention flipped 7 greedy tokens to ` null`. Three of those were matched absent cases, and their paired present cases stayed on the quote:

| Donor | Absent gap after patch | Present gap after patch |
| --- | ---: | ---: |
| `insured_name-7` "not provided" | +0.125 | −0.375 |
| `policy_number-4` "not provided" | +0.125 | −0.375 |
| `insured_name-7` literal `null` | +0.250 | −0.250 |

The present cases that "survived" are a third of a logit from flipping. Across all 120 matched layer-10 pairs, the present case moved toward null by more than the absent case 98 times. A policy-number "not provided" activation, patched into a different field, flipped a present insured name to ` null`. The same happened for the literal-`null` copying control.

Layer 13 feed-forward flipped nothing. Self-patch and final-residual checks reproduced logits exactly. Parameter digest before and after:

`65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`

The measured portion took 26.2 seconds.

## Decision

No weight edit. The only flips sit at layer 10 attention, on two candidate cases plus one copying control, with the present value left just short of erasure. The same intervention is larger on present values than on missing ones, and a donor from the wrong field can delete a real name. That is not a missing-field repair, and it does not identify a sparse set of weights.

## Reproduction

`python scripts/probe_null_token_race.py`. Outputs are in `experiments/null_pairs/token-race-v1/`. The script refuses to overwrite that directory.

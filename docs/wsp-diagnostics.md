# WSP diagnostics — 8 November 2026

No model weights have been changed. Before/after SHA256 digests over all named model parameters match in the completed diagnostic runs. The 49 held-out examples were not accessed. The user requested notice before weight modifications.

## First-token probe

`scripts/probe_fieldbench_start.py` reuses the pinned checkpoint and exact development prompts. It runs one greedy generation step on all 200 development cases, records the top five token scores, and captures the final hidden vector entering the output projection. It verifies prompt hashes against the baseline and parameter integrity before/after inference.

All 200 first tokens match the saved baseline: token 26298, the Markdown fence. The runner-up is token 3219 (`{` followed by newline) in 198 cases, a JSON opening with a quote in one case, and `JSON` in one case. The winner-to-runner-up logit gap is 6.25–11.25, median 7.875. The standalone `{` token ranks 6–19, but that does not represent all valid JSON openings; tokenizer alternatives must be considered.

This is observational evidence for an output-format preference, not causal evidence for a weight edit. Input and output embeddings share their weight tensor. Modifying a fence-token embedding row would therefore affect both generation and interpretation of that token in later inputs. It could also impair legitimate code or Markdown tasks.

## Frozen-weight prefix diagnostic

`scripts/diagnose_json_prefix.py` selects ten development examples per category/source stratum by SHA256 of case ID. This gives 30 cases, including all ten real-source cases; it is not representative of the full corpus distribution. It compares original generation against generation after supplying the single JSON-opening token 3219. This is a diagnostic prompt intervention, explicitly not a WSP solution or a measured benefit of edited weights.

Both arms use BF16, greedy decoding, a total output budget of 768 tokens (767 generated plus the supplied token in the prefix arm), length-sorted left-padded batches, and the same checkpoint and prompts. Parameter hashes are checked before and after. Results are persisted per case.

Batch-size-four results:

| Measure | Original | Supplied JSON opening |
| --- | ---: | ---: |
| Strict raw JSON validity | 0/30 | 28/30 |
| JSON validity after fence removal | 29/30 | 28/30 |
| Schema validity after fence removal | 2/30 | 1/30 |
| Exact field F1 | 21.4% | 24.1% |
| Non-null predictions on gold-null fields | 82/102 | 86/102 |
| Fully correct cases | 0/30 | 0/30 |
| Output token cap reached | 0 | 1 |

The prefix-arm cap hit was `synth_loss_run_015`. Malformed/unparseable outputs contribute no extracted fields under the existing scorer, so false-value rates must be read alongside parse validity, recall, and required-field/schema validity. Missing null keys do not count as hallucinated values but can fail the schema.

The results do not justify selecting a fence-suppression weight edit as a repair for extraction. Formatting can improve while schema correctness and null behavior remain poor. Small sample differences are exploratory, not statistically established improvements or regressions.

## GPU batching

The original arm processed 30 cases in 25.11 seconds with batch size four, versus 57.79 seconds of saved sequential generation time for the same IDs: an observed 2.30x throughput speedup. Peak allocated CUDA memory was 1.26 GiB. These timings exclude model loading and are not individual request latency.

Only 18/30 original-arm strings exactly matched the saved sequential baseline. Batch shape, padding, and numerical execution can affect generation; the cause has not been isolated. Future edits must therefore be compared against a fresh original-model baseline with identical batch composition, ordering, dtype, and decoding settings. Do not attribute differences from the old sequential baseline to WSP.

A second paired run used batch size 16 on the same 30 examples:

| Batch size | Original generation time | Prefix generation time | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: |
| 4 | 25.11 s | 35.81 s | 1.26 GiB |
| 16 | 24.82 s | 27.70 s | 3.20 GiB |

Batch size 16 was only 1.2% faster on the original arm in this single timing comparison; this does not establish it as an optimal batch size. Different generated lengths also affect throughput. It reproduced the qualitative diagnostic result: strict JSON 0/30 versus 28/30; normalized schema validity 2/30 versus 1/30; unsupported values 81/102 versus 83/102; exact field F1 22.36% versus 22.43%. Original outputs matched the saved sequential baseline in 22/30 cases. Parameter integrity hashes matched again. The second run is at `experiments/fieldbench/prefix-diagnostic-batch16-v1/`. Use `--batch-size` and `--run-id` to configure subsequent runs without overwriting results.

## Decision and next experiment

Do not apply a fence-token weight edit on this evidence. First build matched development examples where one explicitly supported value is removed while other valid fields remain, then trace the null-versus-value decision in the hybrid model. Candidate selection should include activation interventions, required-key and retained-value checks, and unrelated tasks including legitimate code fences. Prepare the exact tensor edit and regression plan for the user before applying it. Keep the original checkpoint immutable and the held-out split unused for edit selection.

Raw run directories (ignored by Git): `experiments/fieldbench/start-token-probe-v1/` and `experiments/fieldbench/prefix-diagnostic-batch4-v2/`. An initial batching attempt stopped before generation due to a tokenizer return-type mismatch; it is not an experiment result.

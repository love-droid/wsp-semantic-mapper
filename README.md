# Weight Space Programming for LFM2.5-350M

This repository studies whether small, localized edits to an existing language model's weights can improve insurance-document extraction. The target is LiquidAI/LFM2.5-350M. Experiments use direct edits to existing tensors; they do not use fine-tuning, LoRA, backpropagation, distillation, or architecture changes.

## Main result

The kept experiment improves `form_type` extraction by adding a gated readout to four existing units in the final MLP. The edit changes 12,284 BF16 parameter values across three tensors and adds no parameters.

- Development claim-form errors: 61/61 to 27/61.
- Held-out claim-form errors: 30/30 to 15/30.
- Development field F1 across 200 FieldBench records: 0.2786 to 0.3102.
- Held-out field F1 across 49 records: 0.3547 to 0.3883.
- Unrelated exact answers: 70/100 to 69/100, exactly at the project's 1 percentage-point regression limit.
- Array row F1 and null-array accuracy: unchanged.

This is a targeted form-type result, not a general extraction improvement. Schema validity remained about 4.1% on held-out data, and missing required fields and date formatting remain open problems. The held-out set is small, and five candidate edits were evaluated before selecting the kept edit; see the full record for limitations.

## Checkpoint and patch

The original checkpoint is pinned to revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`. Model files and experiment outputs are intentionally excluded from Git because of size and data privacy. Download the pinned checkpoint using Hugging Face Hub:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LiquidAI/LFM2.5-350M",
    revision="9921cd4d82c0e3986887aa4b376f9446729ef8a3",
    local_dir="models/LFM2.5-350M-9921cd4d",
)
PY
```

The saved edit is stored separately at `models/LFM2.5-350M-9921cd4d-wsp-candidate5-l15-gated-units-r3/`, with its `WSP_EDIT.json` manifest. The original model folder is not overwritten. To reconstruct the edited model, use the candidate script and edit manifest documented in `docs/wsp-full-record.md`; compare the recorded SHA256 before using the result.

## Reproduce the evaluation

Install dependencies in an isolated virtual environment. The pinned model and FieldBench data are downloaded separately and are not included in the repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU dependencies and the Vast.ai workflow, see [`docs/vastai-run.md`](docs/vastai-run.md). For the exact data revision, split rules, license notes, and data preparation, see [`docs/fieldbench-evaluation.md`](docs/fieldbench-evaluation.md).

The full evaluation record is in [`docs/wsp-full-record.md`](docs/wsp-full-record.md), with the result table in [`docs/wsp-results-summary.md`](docs/wsp-results-summary.md). It includes the failed candidates, fit/test protocol, causal probes, held-out measurements, and limitations. The saved predictions, controls and model weights are local experiment artifacts and are intentionally not tracked.

## Research constraints

- Keep the original checkpoint immutable; save edited tensors/model folders separately.
- Do not tune edits on the held-out partition. It was scored once after development selection.
- Compare an edit with a fresh original-model run using identical prompts, batching, dtype, decoding settings, and cases.
- Report unchanged-task and array regression checks along with target metrics.
- Treat FieldBench results as the project's scorer results, not published leaderboard scores.
- Review dataset and document source licenses before redistribution. Do not commit private documents, credentials, model weights, predictions, or held-out data.

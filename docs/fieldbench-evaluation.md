# FieldBench supplemental evaluation

FieldBench is a supplemental, schema-guided extraction benchmark. We use its `insurance_policies` and `insurance_claims` categories because their field-level gold labels include explicit nulls and insurance terminology. It does not replace the LawPAK application/quote benchmark: most inputs are clean text, policy examples are synthetic, and the forms and schemas differ from our target.

## Pinned source and license

- Dataset: [`fieldbench/corpus`](https://huggingface.co/datasets/fieldbench/corpus), revision `7247f4204e4a60a80875d30ef06ea54f8006304a`.
- Dataset card version: FieldBench v0.4.2.
- Schemas: [`fieldbench/corpus`](https://github.com/fieldbench/corpus), commit `57cdb82efaa6c0ba856679a56cf63e614fa1959b`.
- The insurance-policy documents are synthetic. The insurance-claims subset contains synthetic documents and public-domain government forms. The corpus card advertises CC BY 4.0, while the datasheet specifies per-source terms; preserve the source metadata and attribution when sharing derived results.
- The corpus reports 1,441 documents across 10 categories. In our pinned revision the selected categories contain 97 synthetic insurance-policy examples and 152 insurance-claim examples (140 synthetic, 12 real/public-form sources).

The corpus authors report that about 90% of all examples are clean-text extraction rather than rendered-page extraction. Their independent annotation audit samples the wider corpus, not specifically these selected categories. Therefore these results are useful for null handling and schema-driven semantic extraction, but do not establish PDF/OCR performance or category-specific label reliability.

## Frozen split

`scripts/prepare_fieldbench.py` selects only the two categories above and creates a deterministic 80/20 development/held-out split within each category × source stratum. The split is based on a fixed salt and document ID, not on labels or model behavior. The pinned dataset revision, schema revision, case IDs, and file hashes are recorded in the ignored `data/external/fieldbench-v0.4.2/split_manifest.json`.

Current split size: 200 development cases and 49 held-out cases. Use the development file for failure analysis and candidate selection. Do not run the held-out file until the candidate edit and its parameters are frozen; then report held-out results once for the final comparison.

## Run the original model on development cases

The rented Vast.ai RTX 3060 instance has the dataset preparation dependencies installed. After copying this repository's scripts and schema files to the instance, run:

```bash
python scripts/run_fieldbench.py \
  --input data/external/fieldbench-v0.4.2/development.jsonl \
  --model-path models/LFM2.5-350M-9921cd4d \
  --device cuda --dtype bfloat16 \
  --max-new-tokens 768 \
  --run-id fieldbench-dev-original
```

Raw model generations are preserved. Metrics distinguish strict JSON/schema validity from a diagnostic score after stripping only an enclosing Markdown JSON fence. Field comparison is exact and schema validation is independent; report results by category and source. These project metrics are supplemental and are not a claim of reproducing FieldBench's canonical scorer.

All inference remains frozen: no optimizer, gradient, fine-tuning, adapters, or architecture changes.

# Unrelated-task and array controls

These checks are the regression gate for a later weight edit. They are not a candidate and they do not use the 49 held-out FieldBench cases. An edit is a regression if either primary number falls by more than 1 percentage point from the frozen original run of the same file, in the same order, with batch size 16, greedy BF16 decoding, and the pinned LFM2.5-350M checkpoint.

## Unrelated tasks

`data/controls/unrelated-v1.jsonl` has 100 original items: 20 additions, 20 regular plurals, 20 exact copies, 20 Python expression values, and 20 markdown code-fence requests. Exact items are correct when the stripped output or its first non-empty line equals the gold text. Fence items are correct when the raw output contains a markdown fence and the requested Python line. One item is 1 percentage point.

## Arrays

`data/controls/arrays-v1.jsonl` has 70 synthetic documents: 30 practice-area lists, 20 coverage rows, 10 insured addresses, and 10 missing practice-area schedules whose gold value is null. Distractor lines are present and are not part of the gold list. Row F1 uses the project array scorer. Null-array accuracy is the fraction of the 10 missing schedules that come back as JSON null with no extra fields. Each of those two rates has its own 1-point bar.

## Run

```bash
python scripts/build_controls.py
python scripts/run_controls.py --suite unrelated --run-id unrelated-original-v1
python scripts/run_controls.py --suite arrays --run-id arrays-original-v1
```

The runners refuse to overwrite an existing run directory and hash the parameters before and after. Held-out data is not read.

## Original-model baseline

Run on a rented RTX 3090 (Vast.ai, 24 GB), PyTorch 2.7.1+cu128, Transformers 5.22.2, BF16, greedy decoding, batch size 16. Parameter SHA256 before and after: `65d62f3e5a8fee69a5aebe45d666fb3a78e2da774ee559289a062dd65072ef5f`, matching the earlier experiments. The held-out set was not read.

| Check | Result |
| --- | ---: |
| Unrelated exact accuracy (80 items) | 50/80 (62.5%) |
| Unrelated fence accuracy (20 items) | 20/20 (100%) |
| Unrelated overall | 70/100 (70.0%) |
| Array row F1 (60 scored arrays) | 0.589 |
| Array exact row-set rate | 0.133 |
| Null-array accuracy (10 items) | 0/10 |

Run directories: `experiments/controls/unrelated-original-v1/` and `experiments/controls/arrays-original-v1/`.

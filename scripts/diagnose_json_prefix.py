"""Batched frozen-weight diagnostic; prefix intervention is NOT a WSP result."""

import argparse
import hashlib
import json
import time
from collections import defaultdict

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import score_case
from probe_fieldbench_start import digest_parameters
from rescore_baseline import parse_fenced_json
from run_fieldbench import ROOT, DATA_DIR, build_prompt, strict_parse, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-id", default="prefix-diagnostic-batch4-v2")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = ROOT / "experiments/fieldbench" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for line in (DATA_DIR / "development.jsonl").read_text().splitlines():
        case = json.loads(line)
        groups[(case["category"], case["source"])].append(case)
    # Fixed sample selected by ID hash, without reference to labels or scores.
    cases = [case for group in groups.values() for case in sorted(group, key=lambda c: hashlib.sha256(c["id"].encode()).hexdigest())[:10]]
    baseline_dir = ROOT / "experiments/fieldbench/fieldbench-dev-original-v2"
    saved = {r["id"]: r for r in map(json.loads, (baseline_dir / "predictions.jsonl").read_text().splitlines())}
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    for case in cases:
        messages, prompt_hash = build_prompt(case)
        assert prompt_hash == saved[case["id"]]["prompt_sha256"]
        case["prompt_ids"] = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
    cases.sort(key=lambda c: len(c["prompt_ids"]))
    prefix = tokenizer.encode("{\n", add_special_tokens=False)
    assert prefix == [3219]
    report = {"diagnostic_only": True, "modified_parameters": 0, "batch_size": args.batch_size,
              "selection": "10 per development category/source, SHA256(id) order",
              "case_ids": [c["id"] for c in cases], "max_new_tokens_total": 768,
              "dtype": "bfloat16", "device": torch.cuda.get_device_name(),
              "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__}
    for arm in ("original", "json_prefix"):
        predictions, scores = [], []
        elapsed, total_tokens = 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        with (output / f"{arm}.jsonl").open("w") as stream, torch.inference_mode():
            for start in range(0, len(cases), args.batch_size):
                batch = cases[start:start + args.batch_size]
                suffix = prefix if arm == "json_prefix" else []
                items = [{"input_ids": c["prompt_ids"] + suffix, "attention_mask": [1] * (len(c["prompt_ids"]) + len(suffix))} for c in batch]
                encoded = tokenizer.pad(items, padding=True, return_tensors="pt").to("cuda")
                width = encoded["input_ids"].shape[1]
                torch.cuda.synchronize()
                begun = time.perf_counter()
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=768-len(suffix), repetition_penalty=1.0, use_cache=True)
                torch.cuda.synchronize()
                duration = time.perf_counter() - begun
                elapsed += duration
                for case, sequence in zip(batch, generated):
                    tokens = sequence[width:].tolist()
                    if tokenizer.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    total_tokens += len(tokens)
                    raw = ("{\n" if suffix else "") + tokenizer.decode(tokens, skip_special_tokens=True)
                    strict, strict_error = strict_parse(raw)
                    parsed, parse_error, removed = parse_fenced_json(raw)
                    validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
                    prediction = {"id": case["id"], "raw_output": raw, "parsed": parsed,
                                  "strict_json_valid": strict is not None, "normalized_json_valid": parsed is not None,
                                  "strict_schema_valid": strict is not None and validator.is_valid(strict),
                                  "markdown_fence_removed": removed, "parse_error": parse_error,
                                  "latency_seconds": duration/len(batch), "generated_tokens": len(tokens),
                                  "matches_saved_baseline": raw == saved[case["id"]]["raw_output"]}
                    score = score_case(case, prediction, validator)
                    score.update({"category": case["category"], "source": case["source"]})
                    predictions.append(prediction)
                    scores.append(score)
                    stream.write(json.dumps({"prediction": prediction, "score": score}) + "\n")
                stream.flush()
                print(f"{arm}: {min(start+args.batch_size,len(cases))}/{len(cases)}, batch {duration:.2f}s", flush=True)
        report[arm] = {"metrics": summarize(scores, predictions), "generation_seconds": elapsed,
                       "generated_tokens_per_second": total_tokens / elapsed,
                       "cases_per_second": len(cases) / elapsed,
                       "peak_allocated_memory_gib": torch.cuda.max_memory_allocated()/1024**3,
                       "saved_baseline_exact_matches": sum(p["matches_saved_baseline"] for p in predictions)}
    after = digest_parameters(model)
    assert before == after
    report.update({"parameter_sha256_before": before, "parameter_sha256_after": after,
                   "interpretation": "Prompt-prefix diagnostic with frozen weights. Not a weight edit, not a benchmark improvement claim; 49 held-out cases untouched. Batch time per case is amortized throughput, not individual request latency."})
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({arm: report[arm] for arm in ("original", "json_prefix")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

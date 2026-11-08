"""Frozen baseline for the unrelated-task and array controls. Never edits weights."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
import time

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import aggregate, score_case
from probe_fieldbench_start import digest_parameters
from rescore_baseline import parse_fenced_json
from run_fieldbench import MODEL_REVISION, ROOT, build_prompt


SYSTEM = "Answer with only the requested text. Do not explain."
BATCH = 16


def score_unrelated(case, raw):
    stripped = raw.strip()
    first = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    if case["kind"] == "fence":
        correct = "```" in raw and case["gold"] in raw
    else:
        correct = first == case["gold"] or stripped == case["gold"]
    return {"id": case["id"], "kind": case["kind"], "correct": correct, "first_line": first, "raw_output": raw}


def generate_all(model, tokenizer, id_lists, max_new):
    outputs = []
    for start in range(0, len(id_lists), BATCH):
        chunk = id_lists[start:start + BATCH]
        encoded = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in chunk],
            padding=True, return_tensors="pt",
        ).to("cuda")
        width = encoded["input_ids"].shape[1]
        sequences = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, use_cache=True,
        )
        for sequence in sequences:
            tokens = sequence[width:].tolist()
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
            outputs.append(tokenizer.decode(tokens, skip_special_tokens=True))
        print(f"generated {min(start + BATCH, len(id_lists))}/{len(id_lists)}", flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("unrelated", "arrays"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    source = ROOT / "data/controls" / ("unrelated-v1.jsonl" if args.suite == "unrelated" else "arrays-v1.jsonl")
    output = ROOT / "experiments/controls" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    prompts = []
    for case in cases:
        if args.suite == "unrelated":
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]
        else:
            messages, _ = build_prompt(case)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        prompts.append(ids if isinstance(ids, list) else ids.tolist())
    started = time.perf_counter()
    max_new = 64 if args.suite == "unrelated" else 384
    with torch.inference_mode():
        raws = generate_all(model, tokenizer, prompts, max_new)
    if args.suite == "unrelated":
        scored = [score_unrelated(case, raw) for case, raw in zip(cases, raws)]
        by_kind = {}
        for kind in ("exact", "fence"):
            group = [row for row in scored if row["kind"] == kind]
            by_kind[kind] = {"cases": len(group), "correct": sum(row["correct"] for row in group)}
        metrics = {
            "suite": "unrelated", "cases": len(scored),
            "exact_accuracy": sum(row["correct"] for row in scored) / len(scored),
            "by_kind": by_kind,
            "regression_rule": "An edit regresses this check if exact_accuracy falls by more than 0.01 versus this run.",
        }
    else:
        scored = []
        for case, raw in zip(cases, raws):
            parsed, error, fence = parse_fenced_json(raw)
            prediction = {"parsed": parsed, "parse_error": error}
            validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
            score = score_case(case, prediction, validator)
            score["raw_output"] = raw
            score["markdown_fence_removed"] = fence
            score["latency_seconds"] = (time.perf_counter() - started) / len(cases)
            scored.append(score)
        metrics = aggregate(scored, len(cases), len({case["group_id"] for case in cases}))
        metrics.pop("interpretation", None)
        null_cases = [score for case, score in zip(cases, scored) if case["expected"].get("practice_areas") is None and "practice_areas" in case["expected"]]
        metrics["null_arrays"] = {
            "cases": len(null_cases),
            "correct_null": sum(score["all_constraints_satisfied"] for score in null_cases),
        }
        metrics["suite"] = "arrays"
        metrics["regression_rule"] = "An edit regresses this check if array row_f1, or null-array accuracy, falls by more than 0.01 versus this run."
    after = digest_parameters(model)
    if before != after:
        raise AssertionError("parameters changed")
    metrics.update({
        "model_revision": MODEL_REVISION, "modified_parameters": 0,
        "parameter_sha256": after, "batch_size": BATCH, "max_new_tokens": max_new,
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "held_out_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output / "scored.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

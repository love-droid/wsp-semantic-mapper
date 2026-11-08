"""Run frozen LFM2 inference on FieldBench insurance development cases."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import score_case
from rescore_baseline import parse_fenced_json


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/external/fieldbench-v0.4.2"
MODEL_ID = "LiquidAI/LFM2.5-350M"
MODEL_REVISION = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"


def strict_parse(raw: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(raw.strip()), None
    except json.JSONDecodeError as error:
        return None, f"{error.msg} at line {error.lineno}, column {error.colno}"


def build_prompt(case: dict) -> tuple[list[dict[str, str]], str]:
    system = (
        "Extract facts from exactly one document into the provided JSON Schema. "
        "Return one JSON object only, without Markdown or explanation. Include every required key. "
        "Use null when the document does not explicitly support a value, including blank forms, "
        "non-applicable fields, and unresolved conflicts. Do not infer from outside knowledge or "
        "from unrelated context. Map synonyms to the schema's canonical field or option only "
        "when the document context supports the mapping. Preserve source wording where requested."
    )
    user = (
        "Extract this document into the target schema.\n\nTARGET JSON SCHEMA:\n"
        + json.dumps(case["json_schema"], ensure_ascii=False, separators=(",", ":"))
        + "\n\nSOURCE DOCUMENT TEXT:\n"
        + case["input"]
        + "\n\nReturn the complete JSON object now."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_hash = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()
    return messages, prompt_hash


def summarize(scores: list[dict], predictions: list[dict]) -> dict:
    tp = sum(item["fields"]["true_positive"] for item in scores)
    fp = sum(item["fields"]["false_positive"] for item in scores)
    fn = sum(item["fields"]["false_negative"] for item in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    null_fields = sum(item["missing_field_hallucination"]["absent_fields"] for item in scores)
    hallucinations = sum(item["missing_field_hallucination"]["unsupported_values"] for item in scores)
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for item in scores:
        by_slice[f"{item['category']}:{item['source']}"].append(item)

    def field_summary(group: list[dict]) -> dict:
        gtp = sum(item["fields"]["true_positive"] for item in group)
        gfp = sum(item["fields"]["false_positive"] for item in group)
        gfn = sum(item["fields"]["false_negative"] for item in group)
        gp = gtp / (gtp + gfp) if gtp + gfp else 0.0
        gr = gtp / (gtp + gfn) if gtp + gfn else 0.0
        absent = sum(item["missing_field_hallucination"]["absent_fields"] for item in group)
        invented = sum(item["missing_field_hallucination"]["unsupported_values"] for item in group)
        return {
            "cases": len(group),
            "schema_validity_rate": sum(item["schema_valid"] for item in group) / len(group) if group else None,
            "all_constraints_satisfied_rate": sum(item["all_constraints_satisfied"] for item in group) / len(group) if group else None,
            "field_value": {"true_positive": gtp, "false_positive": gfp, "false_negative": gfn,
                            "precision": gp, "recall": gr, "f1": 2 * gp * gr / (gp + gr) if gp + gr else 0.0},
            "missing_field_hallucination": {"unsupported_values": invented, "absent_fields": absent,
                                            "false_value_rate": invented / absent if absent else None},
        }

    strict_valid = sum(prediction["strict_json_valid"] for prediction in predictions)
    normalized_valid = sum(prediction["normalized_json_valid"] for prediction in predictions)
    strict_schema_valid = sum(prediction["strict_schema_valid"] for prediction in predictions)
    return {
        "case_count": len(scores),
        "unique_group_count": len({item["group_id"] for item in scores}),
        "strict_json_validity_rate": strict_valid / len(scores) if scores else None,
        "json_validity_after_markdown_fence_removal": normalized_valid / len(scores) if scores else None,
        "markdown_json_fences_removed": sum(item["markdown_fence_removed"] for item in predictions),
        "strict_schema_validity_rate": strict_schema_valid / len(scores) if scores else None,
        "schema_validity_rate_after_fence_removal": sum(item["schema_valid"] for item in scores) / len(scores) if scores else None,
        "all_constraints_satisfied_rate": sum(item["all_constraints_satisfied"] for item in scores) / len(scores) if scores else None,
        "field_value": {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                        "precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
        "missing_field_hallucination": {"unsupported_values": hallucinations, "absent_fields": null_fields,
                                        "false_value_rate": hallucinations / null_fields if null_fields else None},
        "latency_seconds": {"total": sum(item["latency_seconds"] for item in predictions),
                            "mean_per_case": sum(item["latency_seconds"] for item in predictions) / len(predictions) if predictions else None},
        "by_category_and_source": {key: field_summary(group) for key, group in sorted(by_slice.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "development.jsonl")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/LFM2.5-350M-9921cd4d")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    checkpoint = args.model_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Pinned model checkpoint not found: {checkpoint}")
    if not args.input.is_file():
        raise SystemExit(f"Dataset split not found: {args.input}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No evaluation cases found")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if args.dtype == "auto" and device == "cuda" else (torch.float32 if args.dtype == "auto" else dtypes[args.dtype])
    if device == "cpu" and dtype != torch.float32:
        raise SystemExit("Use float32 for CPU inference")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("fieldbench-%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "experiments/fieldbench" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True).to(device).eval()

    predictions, scores = [], []
    predictions_path = output_dir / "predictions.jsonl"
    scores_path = output_dir / "case_scores.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_file, scores_path.open("w", encoding="utf-8") as score_file:
      for index, case in enumerate(cases, start=1):
        messages, prompt_hash = build_prompt(case)
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt_length = encoded["input_ids"].shape[-1]
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens, repetition_penalty=1.0, use_cache=True)
        latency = time.perf_counter() - started
        raw = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        strict_parsed, strict_error = strict_parse(raw)
        parsed, normalized_error, fence_removed = parse_fenced_json(raw)
        validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
        strict_schema_valid = strict_parsed is not None and not list(validator.iter_errors(strict_parsed))
        prediction = {
            "id": case["id"], "group_id": case["group_id"], "category": case["category"], "source": case["source"],
            "schema_name": case["schema_name"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "prompt_sha256": prompt_hash, "raw_output": raw, "parsed": parsed,
            "strict_parse_error": strict_error, "normalized_parse_error": normalized_error,
            "strict_json_valid": strict_parsed is not None, "normalized_json_valid": parsed is not None,
            "markdown_fence_removed": fence_removed, "strict_schema_valid": strict_schema_valid, "latency_seconds": latency,
            "generated_tokens": int(generated.shape[-1] - prompt_length),
        }
        score = score_case(case, prediction, validator)
        score.update({"category": case["category"], "source": case["source"], "strict_schema_valid": strict_schema_valid,
                      "strict_json_valid": strict_parsed is not None, "markdown_fence_removed": fence_removed,
                      "latency_seconds": latency})
        predictions.append(prediction)
        scores.append(score)
        prediction_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        score_file.write(json.dumps(score, ensure_ascii=False) + "\n")
        prediction_file.flush()
        score_file.flush()
        print(f"[{index}/{len(cases)}] {case['id']}: {latency:.2f}s; strict_json={prediction['strict_json_valid']}; normalized_schema={score['schema_valid']}", flush=True)

    metrics = summarize(scores, predictions)
    metrics["interpretation"] = "Development partition only. FieldBench documents are not an insurance-application benchmark; results are supplemental and must be reported by category/source. Held-out partition was not run."
    metrics["human_reviewed_annotations"] = False
    metrics["modified_parameters"] = 0
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = DATA_DIR / "split_manifest.json"
    run_info = {
        "run_id": run_id, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "checkpoint_path": str(checkpoint / "model.safetensors"), "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_checkpoint_dtype": "BF16", "inference_dtype": str(dtype).removeprefix("torch."), "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "dataset_id": "fieldbench/corpus", "dataset_revision": "7247f4204e4a60a80875d30ef06ea54f8006304a",
        "split": "development", "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "case_count": len(cases),
        "max_new_tokens": args.max_new_tokens, "sampling": False, "repetition_penalty": 1.0,
        "modified_parameters": 0, "gradient_updates": 0, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")
    print(json.dumps({key: value for key, value in metrics.items() if key != "by_category_and_source"}, ensure_ascii=False, indent=2))
    print(json.dumps(metrics["by_category_and_source"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

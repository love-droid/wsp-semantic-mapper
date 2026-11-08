"""Run a frozen, local-only LFM2.5-350M baseline on the local seed set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import aggregate, score_case


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"
CHECKPOINT = ROOT / "models/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots" / REVISION
SCHEMA_PATH = ROOT / "schemas/insurance_submission.v1.schema.json"
DEFAULT_INPUT = ROOT / "data/local/development.seed.jsonl"


def object_from_response(text: str) -> tuple[object | None, str | None]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return None, f"{error.msg} at line {error.lineno}, column {error.colno}"
    return parsed, None


def build_prompt(case: dict, schema: dict) -> tuple[list[dict[str, str]], str]:
    system = (
        "You extract facts from ONE source document into the supplied JSON Schema. "
        "Return only one JSON object: no prose or Markdown. Include every required key. "
        "Use null when a value is missing, ambiguous, or conflicting; do not invent values. "
        "Use only evidence in this document, not outside or companion documents. "
        "Keep arrays in source order, preserve row bases and field roles, and include each row once. "
        "Do not turn a blank into zero or false. Do not infer currency from a dollar sign alone. "
        "Normalize dates only when the date convention is established by the document."
    )
    user = (
        "Extract this document into the target schema.\n\n"
        "TARGET JSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":")) +
        "\n\nSOURCE DOCUMENT TEXT:\n" + case["input"] +
        "\n\nOutput the complete JSON object now."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_hash = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()
    return messages, prompt_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--model-path", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    checkpoint = args.model_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Pinned model checkpoint not found: {checkpoint}")
    if not args.input.is_file():
        raise SystemExit(f"Input dataset not found: {args.input}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise SystemExit(f"No cases found in {args.input}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "experiments/baseline" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
    else:
        dtype = dtype_by_name[args.dtype]
    if device == "cpu" and dtype != torch.float32:
        raise SystemExit("Use float32 for CPU inference; reduced precision is configured for CUDA")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    predictions = []
    case_scores = []
    for index, case in enumerate(cases, start=1):
        messages, prompt_hash = build_prompt(case, schema)
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.0,
                use_cache=True,
            )
        latency = time.perf_counter() - started
        raw = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        parsed, parse_error = object_from_response(raw)
        prediction = {
            "id": case["id"],
            "group_id": case["group_id"],
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "schema": case["schema"],
            "prompt_sha256": prompt_hash,
            "raw_output": raw,
            "parsed": parsed,
            "parse_error": parse_error,
            "latency_seconds": latency,
            "generated_tokens": int(generated.shape[-1] - prompt_length),
        }
        predictions.append(prediction)
        score = score_case(case, prediction, validator)
        score["latency_seconds"] = latency
        case_scores.append(score)
        print(f"[{index}/{len(cases)}] {case['id']}: {latency:.1f}s; JSON={'ok' if parsed is not None else 'invalid'}; schema={'valid' if score['schema_valid'] else 'invalid'}", flush=True)

    aggregate_metrics = aggregate(
        case_scores,
        len(cases),
        len({case["group_id"] for case in cases}),
    )
    aggregate_metrics["case_scores"] = case_scores
    aggregate_metrics["human_reviewed_annotations"] = all(case.get("human_reviewed", False) for case in cases)
    aggregate_metrics["unchanged_task_accuracy_measured"] = False
    aggregate_metrics["modified_parameters"] = 0
    (output_dir / "metrics.json").write_text(json.dumps(aggregate_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(prediction, ensure_ascii=False) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    run_info = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "checkpoint_path": str(checkpoint / "model.safetensors"),
        "model_parameters": parameter_count,
        "source_checkpoint_dtype": "BF16",
        "inference_dtype": str(dtype).removeprefix("torch."),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "modified_parameters": 0,
        "gradient_updates": 0,
        "sampling": False,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "requested_device": args.device,
        "requested_dtype": args.dtype,
        "torch_threads": torch.get_num_threads(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "jsonschema_version": __import__("jsonschema").__version__,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        "human_reviewed_annotations": all(case.get("human_reviewed", False) for case in cases),
        "group_ids": sorted({case["group_id"] for case in cases}),
        "split": "exploratory-development-only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": ["predictions.jsonl", "metrics.json"],
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")
    print(json.dumps({key: value for key, value in aggregate_metrics.items() if key != "case_scores"}, indent=2))


if __name__ == "__main__":
    main()

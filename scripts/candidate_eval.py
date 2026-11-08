"""Shared evaluation for weight-edit candidates: form_type decision, development FieldBench, controls."""
import json
import time
from collections import Counter

import torch
from jsonschema import Draft202012Validator, FormatChecker

from evaluate import aggregate, score_case
from probe_form_type import LABELS
from rescore_baseline import parse_fenced_json
from run_controls import SYSTEM, generate_all, score_unrelated
from run_fieldbench import ROOT, build_prompt, strict_parse, summarize


BATCH = 8
GREEDY_MAX_NEW = 16
FIELDBENCH_MAX_NEW = 768


def pad(tokenizer, id_lists):
    encoded = tokenizer.pad([{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists],
                            padding=True, return_tensors="pt").to("cuda")
    positions = encoded["attention_mask"].long().cumsum(-1) - 1
    positions.masked_fill_(encoded["attention_mask"] == 0, 0)
    return encoded, positions


def decision_choices(model, tokenizer, records, first_token):
    choices = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            chunk = records[start:start + BATCH]
            encoded, positions = pad(tokenizer, [record["ids"] for record in chunk])
            logits = model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            choices.extend(max(LABELS, key=lambda label: float(row[first_token[label]])) for row in logits)
    return choices


def evaluate_all(model, tokenizer, tag, run_dir, dev, fit, test, decision_records, first_token):
    out = {}
    choices = decision_choices(model, tokenizer, decision_records, first_token)
    greedy = generate_all(model, tokenizer, [record["ids"] for record in decision_records], GREEDY_MAX_NEW)
    rows = []
    for record, choice, text in zip(decision_records, choices, greedy):
        rows.append({"id": record["id"], "split": record["split"], "arm": record["arm"], "gold": record["gold"],
                     "choice": choice, "greedy": text, "greedy_exact": text.split('"')[0] == record["gold"],
                     "first_key_choice": choice == record["first_key"]})
    (run_dir / f"{tag}_decision.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    out["decision"] = {}
    for split_name in ("fit", "test"):
        for arm in ("original", "reversed_mapping"):
            group = [row for row in rows if row["split"] == split_name and row["arm"] == arm]
            out["decision"][f"{split_name}:{arm}"] = {
                "label_accuracy": sum(row["choice"] == row["gold"] for row in group) / len(group),
                "first_key_choice": sum(row["first_key_choice"] for row in group) / len(group),
                "greedy_exact": sum(row["greedy_exact"] for row in group) / len(group),
                "greedy_labels": Counter(row["greedy"].split('"')[0] for row in group).most_common(6),
            }

    prompts = []
    for case in dev:
        messages, _ = build_prompt(case)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        prompts.append(ids if isinstance(ids, list) else ids.tolist())
    begin = time.perf_counter()
    with torch.inference_mode():
        raws = generate_all(model, tokenizer, prompts, FIELDBENCH_MAX_NEW)
    latency = (time.perf_counter() - begin) / len(dev)
    predictions, scores = [], []
    for case, raw in zip(dev, raws):
        strict_parsed, strict_error = strict_parse(raw)
        parsed, normalized_error, fence_removed = parse_fenced_json(raw)
        validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
        strict_schema_valid = strict_parsed is not None and not list(validator.iter_errors(strict_parsed))
        prediction = {"id": case["id"], "schema_name": case["schema_name"], "raw_output": raw, "parsed": parsed,
                      "strict_parse_error": strict_error, "normalized_parse_error": normalized_error,
                      "strict_json_valid": strict_parsed is not None, "normalized_json_valid": parsed is not None,
                      "markdown_fence_removed": fence_removed, "strict_schema_valid": strict_schema_valid,
                      "latency_seconds": latency}
        score = score_case(case, prediction, validator)
        score.update({"category": case["category"], "source": case["source"], "strict_schema_valid": strict_schema_valid,
                      "strict_json_valid": strict_parsed is not None, "markdown_fence_removed": fence_removed,
                      "latency_seconds": latency})
        predictions.append(prediction)
        scores.append(score)
    (run_dir / f"{tag}_fieldbench_predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
    (run_dir / f"{tag}_fieldbench_scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))
    metrics = summarize(scores, predictions)
    form_type = {}
    for split_name, ids in (("test", {case["id"] for case in test}), ("fit", {case["id"] for case in fit})):
        group = [(case, prediction) for case, prediction in zip(dev, predictions) if case["id"] in ids]
        errors = sum(not (isinstance(prediction["parsed"], dict) and prediction["parsed"].get("form_type") == case["expected"]["form_type"])
                     for case, prediction in group)
        emitted = sum(isinstance(prediction["parsed"], dict) and "form_type" in prediction["parsed"] for _, prediction in group)
        form_type[split_name] = {"cases": len(group), "errors": errors, "form_type_key_emitted": emitted,
                                 "values": Counter(str(prediction["parsed"].get("form_type")) if isinstance(prediction["parsed"], dict) else "<noparse>"
                                                   for _, prediction in group).most_common(8)}
    out["fieldbench"] = {"field_f1": metrics["field_value"]["f1"], "field_value": metrics["field_value"],
                         "schema_validity": metrics["schema_validity_rate_after_fence_removal"],
                         "json_validity": metrics["json_validity_after_markdown_fence_removal"],
                         "unsupported_values": metrics["missing_field_hallucination"]["unsupported_values"],
                         "form_type": form_type}

    unrelated = [json.loads(line) for line in (ROOT / "data/controls/unrelated-v1.jsonl").open()]
    ids = []
    for case in unrelated:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]
        built = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        ids.append(built if isinstance(built, list) else built.tolist())
    with torch.inference_mode():
        raws = generate_all(model, tokenizer, ids, 64)
    scored = [score_unrelated(case, raw) for case, raw in zip(unrelated, raws)]
    (run_dir / f"{tag}_unrelated.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
    out["unrelated_exact_accuracy"] = sum(row["correct"] for row in scored) / len(scored)

    arrays = [json.loads(line) for line in (ROOT / "data/controls/arrays-v1.jsonl").open()]
    ids = []
    for case in arrays:
        messages, _ = build_prompt(case)
        built = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        ids.append(built if isinstance(built, list) else built.tolist())
    with torch.inference_mode():
        raws = generate_all(model, tokenizer, ids, 384)
    scored = []
    for case, raw in zip(arrays, raws):
        parsed, error, fence = parse_fenced_json(raw)
        validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
        score = score_case(case, {"parsed": parsed, "parse_error": error}, validator)
        score.update({"raw_output": raw, "markdown_fence_removed": fence, "latency_seconds": 0.0})
        scored.append(score)
    (run_dir / f"{tag}_arrays.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
    array_metrics = aggregate(scored, len(arrays), len({case["group_id"] for case in arrays}))
    null_cases = [row for case, row in zip(arrays, scored)
                  if "practice_areas" in case["expected"] and case["expected"]["practice_areas"] is None]
    out["array_row_f1"] = array_metrics["arrays"]["row_f1"]
    out["null_array_accuracy"] = sum(row["all_constraints_satisfied"] for row in null_cases) / len(null_cases)
    print(tag, json.dumps({"decision": {key: value["label_accuracy"] for key, value in out["decision"].items()},
                           "form_type": {key: value["errors"] for key, value in form_type.items()},
                           "field_f1": out["fieldbench"]["field_f1"], "unrelated": out["unrelated_exact_accuracy"],
                           "array_row_f1": out["array_row_f1"], "null_arrays": out["null_array_accuracy"]}), flush=True)
    return out


def pass_checks(original, edited):
    def drop(key):
        return round(original[key] - edited[key], 9)

    before = original["fieldbench"]["form_type"]["test"]["errors"]
    after = edited["fieldbench"]["form_type"]["test"]["errors"]
    checks = {
        "target_relative_reduction": (before - after) / before if before else None,
        "target_pass": before > 0 and after <= 0.9 * before,
        "reversed_arm_pass": edited["decision"]["test:reversed_mapping"]["label_accuracy"] >= original["decision"]["test:reversed_mapping"]["label_accuracy"],
        "field_f1_pass": edited["fieldbench"]["field_f1"] >= original["fieldbench"]["field_f1"],
        "unrelated_drop": drop("unrelated_exact_accuracy"),
        "unrelated_pass": drop("unrelated_exact_accuracy") <= 0.01,
        "array_row_f1_drop": drop("array_row_f1"),
        "array_pass": drop("array_row_f1") <= 0.01 and drop("null_array_accuracy") <= 0.01,
    }
    checks["all_pass"] = all(checks[key] for key in ("target_pass", "reversed_arm_pass", "field_f1_pass", "unrelated_pass", "array_pass"))
    return checks

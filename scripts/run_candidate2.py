"""Candidate 2: sparse layer-15 w2 column edit for the form_type label decision.

Tensor model.layers.15.feed_forward.w2.weight, 64 input columns, 65,536 entries. Closed-form
ridge least squares, no gradients. Columns and the gain/protection setting are chosen on the
61-case fit half only. Caps: at most 10% of the whole tensor and at most 2.5% change in the
protected-position outputs. The edit lives in GPU memory and is reverted before exit.
Development cases only.
"""
import argparse
import json
import random
import time
from collections import Counter

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import aggregate, score_case
from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from rescore_baseline import parse_fenced_json
from run_candidate1 import split_cases
from run_controls import SYSTEM, generate_all, score_unrelated
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt, strict_parse, summarize


LAYER = 15
UNITS = 64
BATCH = 8
PROTECT_PER_PROMPT = 64
GAINS = (4.0, 8.0, 12.0)
MUS = (1.0, 10.0)
MAX_TENSOR_FRACTION = 0.10
MAX_PROTECTED_CHANGE = 0.025
GREEDY_MAX_NEW = 16
FIELDBENCH_MAX_NEW = 768
RUN_DIR = ROOT / "experiments/candidates/candidate2-l15w2-columns"
RULES = {
    "tensor": "model.layers.15.feed_forward.w2.weight, 64 columns, 1024 x 64 = 65,536 entries",
    "construction": "Same fit as candidate2-dryrun-fit-only: units that separate gold form type at the label decision and are quiet elsewhere; Delta solves a ridge push of `gain` logits toward the gold label token, penalised by protected-position outputs.",
    "caps": {"relative_frobenius_vs_tensor": MAX_TENSOR_FRACTION, "protected_output_relative_change": MAX_PROTECTED_CHANGE},
    "selection": "Highest fit-half first-token label accuracy within the caps. Ties take the smaller tensor change. Test half is not used.",
    "split": "Same sha256(id) 61/61 split as candidate 1.",
    "pass": "Test form_type errors fall >= 10% relative. Reversed-mapping test label accuracy does not fall. Development field F1 does not fall. Unrelated accuracy and array row F1 and null-array accuracy each fall by at most 0.01.",
    "held_out": "Not loaded.",
    "weights_on_disk": "Unchanged.",
}


def main():
    global RUN_DIR, GAINS, MUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain", type=float, help="Fix one gain instead of the grid.")
    parser.add_argument("--mu", type=float, help="Fix one protection weight instead of the grid.")
    parser.add_argument("--run-name", default=RUN_DIR.name)
    args = parser.parse_args()
    if (args.gain is None) != (args.mu is None):
        raise SystemExit("--gain and --mu go together")
    if args.gain is not None:
        GAINS, MUS = (args.gain,), (args.mu,)
        RULES["selection"] = f"Fixed before the run: gain {args.gain}, mu {args.mu}. No search. Caps still apply."
    RUN_DIR = RUN_DIR.parent / args.run_name
    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    claims = [case for case in dev if case["schema_name"] == "claim_form.yaml"]
    if len(dev) != 200 or len(claims) != 122:
        raise AssertionError((len(dev), len(claims)))
    fit, test = split_cases(claims)
    test_ids = {case["id"] for case in test}
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    rng = random.Random(20261108)

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    original_hash = digest_parameters(model)
    print("parameter sha256", original_hash, flush=True)
    w2 = model.model.layers[LAYER].feed_forward.w2
    original_w2 = w2.weight.data.clone()
    gamma = model.model.embedding_norm.weight.float()
    unembed = model.lm_head.weight.float()
    eps = model.model.embedding_norm.variance_epsilon
    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}

    def prompt_ids(case, arm):
        built = case if arm == "original" else reversed_case(case)
        messages, _ = build_prompt(built)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
        return (ids if isinstance(ids, list) else ids.tolist()) + quote_ids, next(iter(mapping))

    fit_records = []
    for arm in ("original", "reversed_mapping"):
        for case in fit:
            ids, first_key = prompt_ids(case, arm)
            fit_records.append({"id": case["id"], "gold": case["expected"]["form_type"], "first_key": first_key, "ids": ids, "arm": arm})

    started = time.perf_counter()
    captured = {}
    hooks = [w2.register_forward_pre_hook(lambda m, a: captured.__setitem__("h", a[0][0].float())),
             model.model.layers[LAYER].register_forward_hook(
                 lambda m, a, o: captured.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))]
    hidden, protect, rms_inv, golds = [], [], [], []
    with torch.inference_mode():
        for record in fit_records:
            model(input_ids=torch.tensor([record["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            h = captured["h"]
            hidden.append(h[-1])
            pool = list(range(0, h.shape[0] - 8))
            for position in rng.sample(pool, PROTECT_PER_PROMPT):
                protect.append(h[position])
            rms_inv.append(float(torch.rsqrt(captured["r"].pow(2).mean() + eps)))
            golds.append(record["gold"])
    for hook in hooks:
        hook.remove()
    hidden = torch.stack(hidden).double()
    protect = torch.stack(protect).double()
    classes = sorted(set(golds))
    means = torch.stack([hidden[[i for i, g in enumerate(golds) if g == c]].mean(0) for c in classes])
    within = torch.stack([hidden[[i for i, g in enumerate(golds) if g == c]].var(0) for c in classes]).mean(0)
    between = means.var(0)
    score = between / (within + protect.pow(2).mean(0) + 1e-6)
    units = torch.topk(score, UNITS).indices.sort().values
    expected_units = json.loads((ROOT / "experiments/candidates/candidate2-dryrun-fit-only/summary.json").read_text())["units"]
    if units.tolist() != expected_units:
        raise AssertionError("fit units differ from the dry run")
    h_k = hidden[:, units]
    p_k = protect[:, units]
    directions = {}
    for label in LABELS:
        others = torch.stack([unembed[first_token[o]] for o in LABELS if o != label]).mean(0)
        directions[label] = (gamma * (unembed[first_token[label]] - others))
        directions[label] = (directions[label] / directions[label].pow(2).sum()).double()

    def solve(gain, mu):
        targets = torch.stack([gain / rms_inv[i] * directions[golds[i]] for i in range(len(golds))])
        gram = h_k.T @ h_k
        second = p_k.T @ p_k / len(p_k) * len(h_k)
        system = gram + mu * second + 1e-6 * torch.trace(second) / UNITS * torch.eye(UNITS, dtype=torch.float64, device="cuda")
        return torch.linalg.solve(system, (targets.T @ h_k).T).T

    def pad(id_lists):
        encoded = tokenizer.pad([{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists],
                                padding=True, return_tensors="pt").to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return encoded, positions

    def decision_choices(records):
        choices = []
        with torch.inference_mode():
            for start in range(0, len(records), BATCH):
                chunk = records[start:start + BATCH]
                encoded, positions = pad([record["ids"] for record in chunk])
                logits = model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                choices.extend(max(LABELS, key=lambda label: float(row[first_token[label]])) for row in logits)
        return choices

    columns = original_w2[:, units].double()
    full_protect_out = (protect @ original_w2.double().T).norm()
    grid = []
    for gain in GAINS:
        for mu in MUS:
            delta = solve(gain, mu)
            w2.weight.data[:, units] = (columns + delta).to(w2.weight.dtype)
            choices = decision_choices(fit_records)
            w2.weight.data.copy_(original_w2)
            entry = {"gain": gain, "mu": mu,
                     "fit_label_accuracy": sum(choice == record["gold"] for choice, record in zip(choices, fit_records)) / len(fit_records),
                     "relative_frobenius_vs_columns": float(delta.norm() / columns.norm()),
                     "relative_frobenius_vs_tensor": float(delta.norm() / original_w2.double().norm()),
                     "protected_output_relative_change": float((p_k @ delta.T).norm() / full_protect_out)}
            entry["within_caps"] = (entry["relative_frobenius_vs_tensor"] <= MAX_TENSOR_FRACTION
                                    and entry["protected_output_relative_change"] <= MAX_PROTECTED_CHANGE)
            grid.append(entry)
            print("grid", json.dumps(entry), flush=True)
    if digest_parameters(model) != original_hash:
        raise AssertionError("grid restore failed")
    eligible = [entry for entry in grid if entry["within_caps"]]
    if not eligible:
        (RUN_DIR / "verdict.json").write_text(json.dumps({"aborted": "no setting within caps", "grid": grid}, indent=2) + "\n")
        raise SystemExit("aborted: no setting within caps")
    chosen = max(eligible, key=lambda entry: (entry["fit_label_accuracy"], -entry["relative_frobenius_vs_tensor"]))
    delta = solve(chosen["gain"], chosen["mu"])
    print("chosen", json.dumps(chosen), flush=True)

    decision_records = []
    for split_name, group in (("fit", fit), ("test", test)):
        for arm in ("original", "reversed_mapping"):
            for case in group:
                ids, first_key = prompt_ids(case, arm)
                decision_records.append({"id": case["id"], "split": split_name, "arm": arm,
                                         "gold": case["expected"]["form_type"], "first_key": first_key, "ids": ids})

    def evaluate_all(tag):
        out = {}
        choices = decision_choices(decision_records)
        greedy = generate_all(model, tokenizer, [record["ids"] for record in decision_records], GREEDY_MAX_NEW)
        rows = []
        for record, choice, text in zip(decision_records, choices, greedy):
            rows.append({"id": record["id"], "split": record["split"], "arm": record["arm"], "gold": record["gold"],
                         "choice": choice, "greedy": text, "greedy_exact": text.split('"')[0] == record["gold"],
                         "first_key_choice": choice == record["first_key"]})
        (RUN_DIR / f"{tag}_decision.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
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
        (RUN_DIR / f"{tag}_fieldbench_predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
        (RUN_DIR / f"{tag}_fieldbench_scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))
        metrics = summarize(scores, predictions)
        form_type = {}
        for split_name, ids in (("test", test_ids), ("fit", {case["id"] for case in fit})):
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
        (RUN_DIR / f"{tag}_unrelated.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
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
        (RUN_DIR / f"{tag}_arrays.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scored))
        array_metrics = aggregate(scored, len(arrays), len({case["group_id"] for case in arrays}))
        null_cases = [row for case, row in zip(arrays, scored)
                      if "practice_areas" in case["expected"] and case["expected"]["practice_areas"] is None]
        out["array_row_f1"] = array_metrics["arrays"]["row_f1"]
        out["null_array_accuracy"] = sum(row["all_constraints_satisfied"] for row in null_cases) / len(null_cases)
        print(tag, json.dumps({"decision": {key: value["label_accuracy"] for key, value in out["decision"].items()},
                               "greedy": {key: value["greedy_exact"] for key, value in out["decision"].items()},
                               "form_type": {key: value["errors"] for key, value in form_type.items()},
                               "field_f1": out["fieldbench"]["field_f1"], "unrelated": out["unrelated_exact_accuracy"],
                               "array_row_f1": out["array_row_f1"], "null_arrays": out["null_array_accuracy"]}), flush=True)
        return out

    original = evaluate_all("original")
    if digest_parameters(model) != original_hash:
        raise AssertionError("original evaluation changed parameters")
    w2.weight.data[:, units] = (columns + delta).to(w2.weight.dtype)
    edited_hash = digest_parameters(model)
    changed = w2.weight.data != original_w2
    outside = changed.clone()
    outside[:, units] = False
    if bool(outside.any()):
        raise AssertionError("entries outside the 64 columns changed")
    applied = w2.weight.data[:, units].double() - columns
    edit_info = {
        "tensor": "model.layers.15.feed_forward.w2.weight", "columns": units.tolist(),
        "entries_in_slice": int(columns.numel()), "entries_changed_after_bf16": int(changed.sum()),
        "frobenius_applied": float(applied.norm()), "frobenius_tensor_original": float(original_w2.double().norm()),
        "relative_frobenius_vs_tensor": float(applied.norm() / original_w2.double().norm()),
        "relative_frobenius_vs_columns": float(applied.norm() / columns.norm()),
        "max_abs_change": float(applied.abs().max()), "parameter_sha256_edited": edited_hash,
    }
    print("edit", json.dumps(edit_info), flush=True)
    edited = evaluate_all("edited")
    w2.weight.data.copy_(original_w2)
    restored_hash = digest_parameters(model)

    def drop(key):
        return original[key] - edited[key]

    test_before = original["fieldbench"]["form_type"]["test"]["errors"]
    test_after = edited["fieldbench"]["form_type"]["test"]["errors"]
    checks = {
        "target_relative_reduction": (test_before - test_after) / test_before if test_before else None,
        "target_pass": test_before > 0 and test_after <= 0.9 * test_before,
        "reversed_arm_pass": edited["decision"]["test:reversed_mapping"]["label_accuracy"] >= original["decision"]["test:reversed_mapping"]["label_accuracy"],
        "field_f1_pass": edited["fieldbench"]["field_f1"] >= original["fieldbench"]["field_f1"],
        "unrelated_drop": drop("unrelated_exact_accuracy"),
        "unrelated_pass": drop("unrelated_exact_accuracy") <= 0.01,
        "array_row_f1_drop": drop("array_row_f1"),
        "array_pass": drop("array_row_f1") <= 0.01 and drop("null_array_accuracy") <= 0.01,
    }
    checks["all_pass"] = all(checks[key] for key in ("target_pass", "reversed_arm_pass", "field_f1_pass", "unrelated_pass", "array_pass"))
    verdict = {
        "model_revision": MODEL_REVISION, "candidate": RUN_DIR.name,
        "parameter_sha256_original": original_hash, "parameter_sha256_restored": restored_hash,
        "grid": grid, "chosen": chosen, "edit": edit_info,
        "original": original, "edited": edited, "checks": checks,
        "elapsed_seconds": time.perf_counter() - started, "held_out_used": False, "weights_on_disk_modified": False,
    }
    (RUN_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    print("checks", json.dumps(checks, indent=1), flush=True)
    if restored_hash != original_hash:
        raise AssertionError("restore failed")


if __name__ == "__main__":
    main()

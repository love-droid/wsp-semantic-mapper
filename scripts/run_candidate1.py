"""Candidate 1: closed-form query edit for layer-14 head 5 on the form_type label decision.

Tensor model.layers.14.self_attn.q_proj.weight, rows 320..383 (head 5 queries), 65,536 entries,
rank <= 4. No gradients: ridge least squares on class-mean inputs at the label decision token,
penalised by the second moment of head-5 inputs at other prompt positions. The edit lives only in
GPU memory; the checkpoint on disk is never written. Development cases only.
"""
import hashlib
import json
import random
import time
from collections import Counter

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.lfm2.modeling_lfm2 import rotate_half

from evaluate import aggregate, score_case
from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from rescore_baseline import parse_fenced_json
from run_controls import SYSTEM, generate_all, score_unrelated
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt, strict_parse, summarize


LAYER = 14
HEAD = 5
HEAD_DIM = 64
ROWS = slice(HEAD * HEAD_DIM, (HEAD + 1) * HEAD_DIM)
BATCH = 8
GREEDY_MAX_NEW = 16
FIELDBENCH_MAX_NEW = 768
PROTECT_PER_PROMPT = 64
ALPHAS = (0.1, 0.2, 0.25)
MUS = (3.0, 30.0)
MAX_RELATIVE_NORM = 0.25
MAPPING_LABELS = ["Proof of Loss", "Employer's First Report", "Loss Run", "Demand Letter"]
RUN_DIR = ROOT / "experiments/candidates/candidate1-l14h5-query-r2"
RULES = {
    "revision": "r1 (alpha 0.5-2, mu 0.03-3) aborted at the norm cap before any accuracy was computed; smallest setting was 0.42. r2 changes only the grid.",
    "tensor": "model.layers.14.self_attn.q_proj.weight rows 320..383 (head 5), 64 x 1024 = 65,536 entries",
    "rank": "<= 4 (one class-mean input per gold label)",
    "construction": "Delta = Rbar D Nbar^T (Nbar D Nbar^T + mu_eff C + eps I)^-1. Nbar: class-mean operator_norm output at the label decision token. Rbar: class-mean of alpha * (z* - z), z the head-5 pre-norm query, z* the same norm pointing at the gold mapping key's key vector rotated by RoPE to the decision position and divided by the q_layernorm gain. C: second moment of operator_norm outputs at 64 random other positions per fit prompt. mu_eff = mu * tr(Nbar D Nbar^T) / tr(C).",
    "split": "122 development claim_form cases sorted by sha256(id); first 61 fit, last 61 test. Fit uses original and reversed_mapping prompts.",
    "grid": {"alpha": ALPHAS, "mu": MUS},
    "selection": "Highest fit-set first-token label accuracy over both arms among settings with ||Delta||_F / ||W_rows||_F <= 0.25; ties go to the smaller norm. Test set and FieldBench are not used for selection.",
    "abort": "No setting within the norm cap.",
    "evaluation": "Original and edited evaluated in this script on the same GPU and batching: label decision and greedy label on test and fit (both arms); full 200-case development FieldBench (batch 8, left pad, max_new 768); unrelated-v1 and arrays-v1 controls.",
    "pass": {
        "target": "Test-half form_type errors in full FieldBench generation fall by >= 10% relative (missing key counts as error).",
        "reversed_arm": "Test-half reversed_mapping first-token label accuracy does not fall.",
        "field_f1": "Development field F1 does not fall.",
        "unrelated": "exact_accuracy falls by at most 0.01.",
        "arrays": "row_f1 and null-array accuracy each fall by at most 0.01.",
    },
    "held_out": "Not loaded. Run once only if every pass rule holds.",
    "weights_on_disk": "Unchanged.",
}


def split_cases(cases):
    ordered = sorted(cases, key=lambda case: hashlib.sha256(case["id"].encode()).hexdigest())
    return ordered[:61], ordered[61:]


def key_positions(tokenizer, ids):
    pieces = [tokenizer.decode([token]) for token in ids]
    starts, cursor = [], 0
    for piece in pieces:
        starts.append(cursor)
        cursor += len(piece)
    joined = "".join(pieces)
    mark = joined.find(MAPPING_MARK.strip())
    positions = {}
    for label in MAPPING_LABELS:
        at = joined.find('\\"' + label + '\\": [', mark)
        if at < 0:
            raise AssertionError(f"{label} key not found")
        positions[label] = max(position for position, start in enumerate(starts) if start <= at + 2)
    return positions


def main():
    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    claims = [case for case in dev if case["schema_name"] == "claim_form.yaml"]
    if len(dev) != 200 or len(claims) != 122:
        raise AssertionError((len(dev), len(claims)))
    fit, test = split_cases(claims)
    test_ids = {case["id"] for case in test}
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    (RUN_DIR / "split.json").write_text(json.dumps({"fit": [c["id"] for c in fit], "test": [c["id"] for c in test]}, indent=2) + "\n")
    rng = random.Random(20261108)

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    original_hash = digest_parameters(model)
    print("parameter sha256", original_hash, flush=True)
    layer = model.model.layers[LAYER]
    attention = layer.self_attn
    kv_head = HEAD // attention.num_key_value_groups
    weight = attention.q_proj.weight
    original_rows = weight.data[ROWS].clone()
    original_tensor = weight.data.clone()

    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}

    def decision_ids(case):
        messages, _ = build_prompt(case)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        return (ids if isinstance(ids, list) else ids.tolist()) + quote_ids

    decision_records = []
    for split_name, group in (("fit", fit), ("test", test)):
        for arm in ("original", "reversed_mapping"):
            for case in group:
                built = case if arm == "original" else reversed_case(case)
                ids = decision_ids(built)
                mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
                decision_records.append({"id": case["id"], "split": split_name, "arm": arm,
                                         "gold": case["expected"]["form_type"], "first_key": next(iter(mapping)),
                                         "ids": ids, "keys": key_positions(tokenizer, ids)})

    started = time.perf_counter()
    captured = {}
    hooks = [
        layer.operator_norm.register_forward_hook(lambda m, a, o: captured.__setitem__("norm", o[0].float())),
        attention.q_proj.register_forward_hook(lambda m, a, o: captured.__setitem__("q", o[0, :, ROWS].float())),
        attention.k_proj.register_forward_hook(lambda m, a, o: captured.__setitem__("k", o[0].float())),
    ]
    inputs, residual_dirs, query_norms, current, labels, protect = [], [], [], [], [], []
    q_gain = attention.q_layernorm.weight.float()
    with torch.inference_mode():
        for record in [r for r in decision_records if r["split"] == "fit"]:
            ids = torch.tensor([record["ids"]], device="cuda")
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
            length = ids.shape[1]
            p = length - 1
            j = record["keys"][record["gold"]]
            keys = captured["k"].view(length, -1, HEAD_DIM)[:, kv_head]
            key = attention.k_layernorm(keys[j:j + 1].to(torch.bfloat16)).float()
            positions = torch.tensor([[j, p]], device="cuda")
            cos, sin = model.model.rotary_emb(key.unsqueeze(0), positions)
            cos, sin = cos[0].float(), sin[0].float()
            rotated = key * cos[0] + rotate_half(key) * sin[0]
            aligned = rotated * cos[1] - rotate_half(rotated) * sin[1]
            direction = (aligned[0] / q_gain)
            direction = direction / direction.norm()
            z = captured["q"][p]
            inputs.append(captured["norm"][p])
            current.append(z)
            residual_dirs.append(z.norm() * direction - z)
            labels.append(record["gold"])
            pool = list(range(0, max(1, p - 8)))
            for position in rng.sample(pool, min(PROTECT_PER_PROMPT, len(pool))):
                protect.append(captured["norm"][position])
    for hook in hooks:
        hook.remove()
    inputs = torch.stack(inputs).double()
    residual_dirs = torch.stack(residual_dirs).double()
    protect = torch.stack(protect).double()
    classes = sorted(set(labels))
    counts = torch.tensor([labels.count(c) for c in classes], dtype=torch.float64, device="cuda")
    class_inputs = torch.stack([inputs[[i for i, l in enumerate(labels) if l == c]].mean(0) for c in classes])
    class_residuals = torch.stack([residual_dirs[[i for i, l in enumerate(labels) if l == c]].mean(0) for c in classes])
    second_moment = protect.T @ protect / len(protect)
    gram = class_inputs.T @ torch.diag(counts) @ class_inputs
    w_rows = original_rows.double()
    print(f"fit capture done {time.perf_counter() - started:.1f}s; classes {dict(zip(classes, counts.tolist()))}; protect {len(protect)}", flush=True)

    def solve(alpha, mu):
        mu_eff = mu * float(torch.trace(gram) / torch.trace(second_moment))
        eps = 1e-6 * float(torch.trace(second_moment)) / second_moment.shape[0]
        system = gram + mu_eff * second_moment + eps * torch.eye(gram.shape[0], dtype=torch.float64, device="cuda")
        right = alpha * class_residuals.T @ torch.diag(counts) @ class_inputs
        return torch.linalg.solve(system, right.T).T

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
                encoded, positions = pad([r["ids"] for r in chunk])
                logits = model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                for row in logits:
                    choices.append(max(LABELS, key=lambda label: float(row[first_token[label]])))
        return choices

    def set_rows(delta):
        weight.data[ROWS] = (w_rows + delta).to(weight.dtype) if delta is not None else original_rows

    fit_records = [r for r in decision_records if r["split"] == "fit"]
    grid = []
    for alpha in ALPHAS:
        for mu in MUS:
            delta = solve(alpha, mu)
            relative = float(delta.norm() / w_rows.norm())
            entry = {"alpha": alpha, "mu": mu, "relative_frobenius": relative, "rank": int(torch.linalg.matrix_rank(delta.float()))}
            if relative <= MAX_RELATIVE_NORM:
                set_rows(delta)
                choices = decision_choices(fit_records)
                entry["fit_label_accuracy"] = sum(c == r["gold"] for c, r in zip(choices, fit_records)) / len(fit_records)
                entry["protected_query_relative_change"] = float((protect @ delta.T).norm() / (protect @ w_rows.T).norm())
                set_rows(None)
            grid.append(entry)
            print("grid", json.dumps(entry), flush=True)
    if digest_parameters(model) != original_hash:
        raise AssertionError("restore failed")
    eligible = [entry for entry in grid if "fit_label_accuracy" in entry]
    if not eligible:
        (RUN_DIR / "verdict.json").write_text(json.dumps({"aborted": "no setting within norm cap", "grid": grid}, indent=2) + "\n")
        raise SystemExit("aborted: no setting within norm cap")
    chosen = max(eligible, key=lambda entry: (entry["fit_label_accuracy"], -entry["relative_frobenius"]))
    delta = solve(chosen["alpha"], chosen["mu"])
    print("chosen", json.dumps(chosen), flush=True)

    def evaluate_all(tag):
        out = {}
        choices = decision_choices(decision_records)
        greedy = generate_all(model, tokenizer, [r["ids"] for r in decision_records], GREEDY_MAX_NEW)
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
        for split_name, ids in (("test", test_ids), ("fit", {c["id"] for c in fit})):
            group = [(case, p) for case, p in zip(dev, predictions) if case["id"] in ids]
            errors = sum(not (isinstance(p["parsed"], dict) and p["parsed"].get("form_type") == case["expected"]["form_type"])
                         for case, p in group)
            emitted = sum(isinstance(p["parsed"], dict) and "form_type" in p["parsed"] for _, p in group)
            form_type[split_name] = {"cases": len(group), "errors": errors, "form_type_key_emitted": emitted,
                                     "values": Counter(str(p["parsed"].get("form_type")) if isinstance(p["parsed"], dict) else "<noparse>"
                                                       for _, p in group).most_common(8)}
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
        null_cases = [s for case, s in zip(arrays, scored)
                      if "practice_areas" in case["expected"] and case["expected"]["practice_areas"] is None]
        out["array_row_f1"] = array_metrics["arrays"]["row_f1"]
        out["null_array_accuracy"] = sum(s["all_constraints_satisfied"] for s in null_cases) / len(null_cases)
        print(tag, json.dumps({"decision": {k: v["label_accuracy"] for k, v in out["decision"].items()},
                               "form_type": {k: v["errors"] for k, v in form_type.items()},
                               "field_f1": out["fieldbench"]["field_f1"], "unrelated": out["unrelated_exact_accuracy"],
                               "array_row_f1": out["array_row_f1"], "null_arrays": out["null_array_accuracy"]}), flush=True)
        return out

    original = evaluate_all("original")
    if digest_parameters(model) != original_hash:
        raise AssertionError("original evaluation changed parameters")
    set_rows(delta)
    edited_hash = digest_parameters(model)
    changed = (weight.data != original_tensor)
    outside = changed.clone()
    outside[ROWS] = False
    if bool(outside.any()):
        raise AssertionError("entries outside rows 320..383 changed")
    applied = weight.data[ROWS].double() - w_rows
    edit_info = {
        "tensor": "model.layers.14.self_attn.q_proj.weight", "rows": [ROWS.start, ROWS.stop - 1],
        "entries_in_slice": int(original_rows.numel()), "entries_changed_after_bf16": int(changed.sum()),
        "frobenius_applied": float(applied.norm()), "frobenius_rows_original": float(w_rows.norm()),
        "relative_frobenius_applied": float(applied.norm() / w_rows.norm()),
        "max_abs_change": float(applied.abs().max()), "rank_requested": int(torch.linalg.matrix_rank(delta.float())),
        "parameter_sha256_edited": edited_hash,
    }
    print("edit", json.dumps(edit_info), flush=True)
    edited = evaluate_all("edited")
    set_rows(None)
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
        "model_revision": MODEL_REVISION, "candidate": "candidate1-l14h5-query",
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

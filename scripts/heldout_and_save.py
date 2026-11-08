"""One-time held-out check for a passing layer-15 w2 candidate, then save it as a new checkpoint.

Rebuilds the edit exactly as run_candidate2 fits it (fit half only) and requires the edited
parameter hash to match the candidate's verdict. Scores the 49 held-out FieldBench cases with the
original and edited weights back to back. Saves the edited model to a new folder only if the
held-out rules pass. The original checkpoint folder is never written.
"""
import argparse
import hashlib
import json
import random
import shutil
import time
from collections import Counter

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import score_case
from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from rescore_baseline import parse_fenced_json
from run_candidate1 import split_cases
from run_controls import generate_all
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt, strict_parse, summarize


LAYER = 15
UNITS = 64
PROTECT_PER_PROMPT = 64
FIELDBENCH_MAX_NEW = 768
HELD_OUT_RULES = {
    "target": "Held-out claim_form form_type errors fall by >= 10% relative (missing key counts as error).",
    "field_f1": "Held-out field F1 does not fall.",
    "runs": "Once. Original and edited on the same GPU and batching.",
    "save": "Only if both rules pass. New folder; original checkpoint untouched.",
}


def fit_edit(model, tokenizer, fit, gain, mu):
    rng = random.Random(20261108)
    w2 = model.model.layers[LAYER].feed_forward.w2
    gamma = model.model.embedding_norm.weight.float()
    unembed = model.lm_head.weight.float()
    eps = model.model.embedding_norm.variance_epsilon
    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}
    records = []
    for arm in ("original", "reversed_mapping"):
        for case in fit:
            built = case if arm == "original" else reversed_case(case)
            messages, _ = build_prompt(built)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
            records.append({"gold": case["expected"]["form_type"],
                            "ids": (ids if isinstance(ids, list) else ids.tolist()) + quote_ids})
    captured = {}
    hooks = [w2.register_forward_pre_hook(lambda m, a: captured.__setitem__("h", a[0][0].float())),
             model.model.layers[LAYER].register_forward_hook(
                 lambda m, a, o: captured.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))]
    hidden, protect, rms_inv, golds = [], [], [], []
    with torch.inference_mode():
        for record in records:
            model(input_ids=torch.tensor([record["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            h = captured["h"]
            hidden.append(h[-1])
            for position in rng.sample(list(range(0, h.shape[0] - 8)), PROTECT_PER_PROMPT):
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
    score = means.var(0) / (within + protect.pow(2).mean(0) + 1e-6)
    units = torch.topk(score, UNITS).indices.sort().values
    h_k, p_k = hidden[:, units], protect[:, units]
    directions = {}
    for label in LABELS:
        others = torch.stack([unembed[first_token[o]] for o in LABELS if o != label]).mean(0)
        e = gamma * (unembed[first_token[label]] - others)
        directions[label] = (e / e.pow(2).sum()).double()
    targets = torch.stack([gain / rms_inv[i] * directions[golds[i]] for i in range(len(golds))])
    second = p_k.T @ p_k / len(p_k) * len(h_k)
    system = h_k.T @ h_k + mu * second + 1e-6 * torch.trace(second) / UNITS * torch.eye(UNITS, dtype=torch.float64, device="cuda")
    delta = torch.linalg.solve(system, (targets.T @ h_k).T).T
    return units, delta


def run_fieldbench(model, tokenizer, cases):
    prompts = []
    for case in cases:
        messages, _ = build_prompt(case)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        prompts.append(ids if isinstance(ids, list) else ids.tolist())
    begin = time.perf_counter()
    with torch.inference_mode():
        raws = generate_all(model, tokenizer, prompts, FIELDBENCH_MAX_NEW)
    latency = (time.perf_counter() - begin) / len(cases)
    predictions, scores = [], []
    for case, raw in zip(cases, raws):
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
    metrics = summarize(scores, predictions)
    claims = [(case, p) for case, p in zip(cases, predictions) if case["schema_name"] == "claim_form.yaml"]
    errors = sum(not (isinstance(p["parsed"], dict) and p["parsed"].get("form_type") == case["expected"]["form_type"])
                 for case, p in claims)
    return predictions, scores, {
        "field_f1": metrics["field_value"]["f1"], "field_value": metrics["field_value"],
        "schema_validity": metrics["schema_validity_rate_after_fence_removal"],
        "claim_cases": len(claims), "form_type_errors": errors,
        "form_type_values": Counter(str(p["parsed"].get("form_type")) if isinstance(p["parsed"], dict) else "<noparse>"
                                    for _, p in claims).most_common(8),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Run folder name under experiments/candidates.")
    args = parser.parse_args()
    candidate_dir = ROOT / "experiments/candidates" / args.candidate
    verdict = json.loads((candidate_dir / "verdict.json").read_text())
    if not verdict["checks"]["all_pass"]:
        raise SystemExit("candidate did not pass development checks")
    run_dir = ROOT / "experiments/heldout" / args.candidate
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "rules.json").write_text(json.dumps(HELD_OUT_RULES, indent=2) + "\n")

    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    fit, _ = split_cases([case for case in dev if case["schema_name"] == "claim_form.yaml"])
    held_out_path = DATA_DIR / "heldout.jsonl"
    held_out = [json.loads(line) for line in held_out_path.open()]
    if len(held_out) != 49:
        raise AssertionError(len(held_out))

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    original_hash = digest_parameters(model)
    if original_hash != verdict["parameter_sha256_original"]:
        raise AssertionError("original checkpoint hash differs from the candidate run")
    w2 = model.model.layers[LAYER].feed_forward.w2
    original_w2 = w2.weight.data.clone()
    chosen = verdict["chosen"]
    saved_edit = candidate_dir / "edit.pt"
    edit = torch.load(saved_edit) if saved_edit.exists() else None
    if edit is not None and "slices" in edit:
        slices = edit["slices"]
    else:
        if edit is not None:
            units = edit["columns"].to("cuda")
            edited_columns = edit["edited_columns"].to("cuda", w2.weight.dtype)
        else:
            units, delta = fit_edit(model, tokenizer, fit, chosen["gain"], chosen["mu"])
            edited_columns = (original_w2[:, units].double() + delta).to(w2.weight.dtype)
        if units.tolist() != verdict["edit"]["columns"]:
            raise AssertionError("rebuilt columns differ")
        slices = [{"tensor": f"model.layers.{LAYER}.feed_forward.w2.weight", "dim": 1, "index": units, "values": edited_columns}]

    original_predictions, original_scores, original = run_fieldbench(model, tokenizer, held_out)
    for piece in slices:
        parameter = model.get_parameter(piece["tensor"])
        index = piece["index"].to("cuda")
        values = piece["values"].to("cuda", parameter.dtype)
        if piece["dim"] == 0:
            parameter.data[index] = values
        else:
            parameter.data[:, index] = values
    edited_hash = digest_parameters(model)
    if edited_hash != verdict["edit"]["parameter_sha256_edited"]:
        raise AssertionError("rebuilt edit hash differs from the candidate run")
    edited_predictions, edited_scores, edited = run_fieldbench(model, tokenizer, held_out)
    for tag, predictions, scores in (("original", original_predictions, original_scores),
                                     ("edited", edited_predictions, edited_scores)):
        (run_dir / f"{tag}_predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
        (run_dir / f"{tag}_scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))

    before, after = original["form_type_errors"], edited["form_type_errors"]
    checks = {
        "target_relative_reduction": (before - after) / before if before else None,
        "target_pass": before > 0 and after <= 0.9 * before,
        "field_f1_pass": edited["field_f1"] >= original["field_f1"],
    }
    checks["all_pass"] = checks["target_pass"] and checks["field_f1_pass"]
    result = {"candidate": args.candidate, "model_revision": MODEL_REVISION,
              "held_out_sha256": hashlib.sha256(held_out_path.read_bytes()).hexdigest(),
              "parameter_sha256_original": original_hash, "parameter_sha256_edited": edited_hash,
              "original": original, "edited": edited, "checks": checks}

    if checks["all_pass"]:
        target = ROOT / f"models/LFM2.5-350M-9921cd4d-wsp-{args.candidate}"
        if target.exists():
            raise AssertionError(f"{target} already exists")
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        for name in ("config.json", "generation_config.json"):
            if (checkpoint / name).exists():
                shutil.copy2(checkpoint / name, target / name)
        reloaded = AutoModelForCausalLM.from_pretrained(target, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
        reloaded_hash = digest_parameters(reloaded)
        if reloaded_hash != edited_hash:
            raise AssertionError("saved checkpoint hash differs from the edited model")
        (target / "WSP_EDIT.json").write_text(json.dumps({
            "base_model": "LiquidAI/LFM2.5-350M", "base_revision": MODEL_REVISION,
            "base_parameter_sha256": original_hash, "edited_parameter_sha256": edited_hash,
            "edit": verdict["edit"], "chosen": chosen, "development_checks": verdict["checks"],
            "held_out_checks": checks, "method": "closed-form ridge least squares, no gradients",
        }, indent=2) + "\n")
        result["saved_to"] = str(target)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"original": {k: original[k] for k in ("field_f1", "form_type_errors", "claim_cases")},
                      "edited": {k: edited[k] for k in ("field_f1", "form_type_errors", "claim_cases")},
                      "checks": checks, "saved_to": result.get("saved_to")}, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""Frozen diagnostic of the claim_form form_type label decision. Never edits weights.

Development FieldBench only. Every claim_form title names its canonical label or a listed
synonym, yet the original run never emits the right form_type. This asks whether the model
ranks the right label low, whether it copies the first label in the mapping list, and which
layer components push toward the wrong label at the label decision token.
"""
import json
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt


BATCH = 8
GREEDY_MAX_NEW = 16
LABELS = ["Proof of Loss", "Employer's First Report", "Loss Run", "Demand Letter", "Other"]
OPTIONS = LABELS + [None]
PREFIX = '```json\n{\n  "form_type":'
MAPPING_MARK = "Canonical label mappings: "
RUN_DIR = ROOT / "experiments/form_type/decision-v1"
RULES = {
    "question": "At the form_type label token, does the model rank the gold label low, copy the first mapping entry, and where does the wrong-label push enter?",
    "cases": "All 122 development claim_form.yaml cases. Held-out not loaded.",
    "arms": {
        "original": "Unchanged prompt.",
        "reversed_mapping": "Same prompt with only the order of the canonical-label mapping dict reversed, so Demand Letter is listed last.",
    },
    "prefix": PREFIX + " then each option. String options are scored as ' \"<label>\"'; null as ' null'.",
    "label_decision": "Position after the opening quote token. First label tokens are distinct.",
    "probe": "Nearest-centroid leave-one-out accuracy on the residual after each layer at the label decision position, original arm, four gold classes.",
    "attribution": "Linear share of each embedding/mixer/ffn output in (gold first token minus Demand first token) using the final RMS scale of that case. Exact up to bf16 rounding.",
    "weights": "Frozen.",
    "stop_rule": "No further prompt arms in this run. A weight candidate is proposed separately before any edit.",
}


def reversed_case(case):
    schema = json.loads(json.dumps(case["json_schema"]))
    field = schema["properties"]["form_type"]
    head, mapping = field["description"].split(MAPPING_MARK)
    ordered = json.loads(mapping)
    field["description"] = head + MAPPING_MARK + json.dumps(dict(reversed(list(ordered.items()))))
    if list(json.loads(field["description"].split(MAPPING_MARK)[1]))[-1] != "Demand Letter":
        raise AssertionError("reversal failed")
    return {**case, "json_schema": schema}


def main():
    cases = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    cases = [case for case in cases if case["schema_name"] == "claim_form.yaml"]
    if len(cases) != 122:
        raise AssertionError(len(cases))
    for case in cases:
        mapping = case["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1]
        if json.dumps(json.loads(mapping)) != mapping:
            raise AssertionError("mapping not re-serialisable for " + case["id"])
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    print("model loaded; hashing parameters", flush=True)
    before = digest_parameters(model)
    print("parameter sha256", before, flush=True)

    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    if quote_ids[:len(prefix_ids)] != prefix_ids or len(quote_ids) != len(prefix_ids) + 1:
        raise AssertionError("opening quote is not one token after the prefix")
    option_tails = {}
    for option in OPTIONS:
        text = PREFIX + (" null" if option is None else ' "' + option + '"')
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids[:len(prefix_ids)] != prefix_ids:
            raise AssertionError(f"prefix not stable for {option}")
        if option is not None and ids[:len(quote_ids)] != quote_ids:
            raise AssertionError(f"quote merges with {option}")
        option_tails[option] = ids[len(prefix_ids):]
    first_token = {label: option_tails[label][1] for label in LABELS}
    if len(set(first_token.values())) != len(LABELS):
        raise AssertionError(f"first label tokens collide {first_token}")

    records = []
    for arm in ("original", "reversed_mapping"):
        for case in cases:
            built = case if arm == "original" else reversed_case(case)
            messages, prompt_hash = build_prompt(built)
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids.tolist()
            records.append({
                "id": case["id"], "arm": arm, "gold": case["expected"]["form_type"],
                "source": case["source"], "prompt_sha256": prompt_hash,
                "prompt_ids": prompt_ids,
            })

    def pad(id_lists):
        encoded = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists],
            padding=True, return_tensors="pt",
        ).to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return encoded, positions

    started = time.perf_counter()
    jobs = [(index, option) for index in range(len(records)) for option in OPTIONS]
    scores = defaultdict(dict)
    longest = max(len(tail) for tail in option_tails.values())
    with torch.inference_mode():
        for start in range(0, len(jobs), BATCH):
            chunk = jobs[start:start + BATCH]
            id_lists = [records[index]["prompt_ids"] + prefix_ids + option_tails[option] for index, option in chunk]
            encoded, positions = pad(id_lists)
            logits = model(**encoded, position_ids=positions, use_cache=False,
                           logits_to_keep=longest + 1).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for row, (index, option) in enumerate(chunk):
                tail = option_tails[option]
                total = 0.0
                for offset, token in enumerate(tail):
                    total += float(logprobs[row, logprobs.shape[1] - 1 - len(tail) + offset, token])
                scores[index][option] = total
    print(f"sequence scoring done {time.perf_counter() - started:.1f}s", flush=True)

    layers = model.model.layers
    captured = {}
    hooks = []

    def keep(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.setdefault(name, []).append(value[:, -1].detach().float().clone())
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(keep("embed")))
    for index, layer in enumerate(layers):
        mixer = layer.self_attn if layer.is_attention_layer else layer.conv
        kind = "attention" if layer.is_attention_layer else "conv"
        hooks.append(mixer.register_forward_hook(keep(f"layer{index}.{kind}")))
        hooks.append(layer.feed_forward.register_forward_hook(keep(f"layer{index}.ffn")))
        hooks.append(layer.register_forward_hook(keep(f"resid{index}")))
    decision_logits = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            chunk = records[start:start + BATCH]
            encoded, positions = pad([record["prompt_ids"] + quote_ids[:] for record in chunk])
            decision_logits.append(model(**encoded, position_ids=positions, use_cache=False,
                                         logits_to_keep=1).logits[:, -1].float())
    for hook in hooks:
        hook.remove()
    decision_logits = torch.cat(decision_logits)
    captured = {name: torch.cat(chunks) for name, chunks in captured.items()}
    print(f"decision capture done {time.perf_counter() - started:.1f}s", flush=True)

    greedy = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            chunk = records[start:start + BATCH]
            encoded, _ = pad([record["prompt_ids"] + quote_ids for record in chunk])
            width = encoded["input_ids"].shape[1]
            out = model.generate(**encoded, do_sample=False, max_new_tokens=GREEDY_MAX_NEW,
                                 repetition_penalty=1.0, use_cache=True)
            greedy.extend(tokenizer.decode(seq[width:], skip_special_tokens=True) for seq in out)

    component_names = ["embed"] + [name for index, layer in enumerate(layers) for name in (
        f"layer{index}.{'attention' if layer.is_attention_layer else 'conv'}", f"layer{index}.ffn")]
    final_resid = captured["resid15"]
    reconstructed = sum(captured[name] for name in component_names)
    reconstruction_error = float((reconstructed - final_resid).abs().max() / final_resid.abs().max())
    norm = model.model.embedding_norm
    head = model.lm_head.weight.float()
    gain = norm.weight.float()
    rms = torch.rsqrt(final_resid.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    lens_logits_check = float((torch.nn.functional.linear(final_resid * rms * gain, head) - decision_logits).abs().max())

    def lens(resid):
        scale = torch.rsqrt(resid.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return torch.nn.functional.linear(resid * scale * gain, head)

    demand = first_token["Demand Letter"]
    rows = []
    attribution = defaultdict(list)
    for index, record in enumerate(records):
        option_scores = scores[index]
        ranked = sorted(OPTIONS, key=lambda option: -option_scores[option])
        gold = record["gold"]
        best_wrong = max((option for option in OPTIONS if option != gold), key=lambda option: option_scores[option])
        label_logits = {label: float(decision_logits[index, first_token[label]]) for label in LABELS}
        lens_gap = []
        for layer_index in range(len(layers)):
            values = lens(captured[f"resid{layer_index}"][index:index + 1])[0]
            lens_gap.append(float(values[first_token[gold]] - values[demand]) if gold != "Demand Letter" else None)
        if gold != "Demand Letter" and record["arm"] == "original":
            direction = (head[first_token[gold]] - head[demand]) * gain * rms[index]
            for name in component_names:
                attribution[name].append(float(captured[name][index] @ direction))
        rows.append({
            "id": record["id"], "arm": record["arm"], "gold": gold, "source": record["source"],
            "prompt_sha256": record["prompt_sha256"],
            "sequence_logprob": {str(option): option_scores[option] for option in OPTIONS},
            "constrained_choice": ranked[0], "gold_rank": ranked.index(gold) + 1,
            "gold_minus_best_wrong": option_scores[gold] - option_scores[best_wrong],
            "best_wrong": best_wrong,
            "decision_top_token": tokenizer.decode([int(decision_logits[index].argmax())]),
            "decision_label_first_token_logits": label_logits,
            "greedy_after_quote": greedy[index],
            "greedy_exact_label": greedy[index].split('"')[0] == gold,
            "logit_lens_gold_minus_demand_by_layer": lens_gap,
        })
    with (RUN_DIR / "records.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    originals = [index for index, record in enumerate(records) if record["arm"] == "original"]
    classes = sorted({records[index]["gold"] for index in originals})
    probe = {}
    for layer_index in range(len(layers)):
        features = captured[f"resid{layer_index}"][originals]
        features = (features - features.mean(0)) / (features.std(0) + 1e-6)
        labels = [records[index]["gold"] for index in originals]
        correct = 0
        for held in range(len(originals)):
            centroids = {}
            for name in classes:
                members = [row for row in range(len(originals)) if row != held and labels[row] == name]
                centroids[name] = features[members].mean(0)
            guess = min(classes, key=lambda name: float((features[held] - centroids[name]).norm()))
            correct += guess == labels[held]
        probe[f"resid{layer_index}"] = correct / len(originals)

    def arm_summary(arm):
        chosen = [row for row in rows if row["arm"] == arm]
        wrong_gold = [row for row in chosen if row["gold"] != "Demand Letter"]
        return {
            "cases": len(chosen),
            "constrained_accuracy": sum(row["constrained_choice"] == row["gold"] for row in chosen) / len(chosen),
            "greedy_exact_label_accuracy": sum(row["greedy_exact_label"] for row in chosen) / len(chosen),
            "constrained_choice_counts": Counter(str(row["constrained_choice"]) for row in chosen),
            "gold_rank_counts": Counter(row["gold_rank"] for row in chosen),
            "confusion": Counter(f"{row['gold']} -> {row['constrained_choice']}" for row in chosen),
            "decision_top_tokens": Counter(row["decision_top_token"] for row in chosen).most_common(10),
            "greedy_label_counts": Counter(row["greedy_after_quote"].split('"')[0] for row in chosen).most_common(12),
            "median_gold_minus_best_wrong": sorted(row["gold_minus_best_wrong"] for row in chosen)[len(chosen) // 2],
            "non_demand_gold_cases": len(wrong_gold),
        }

    mean_attribution = {name: sum(values) / len(values) for name, values in attribution.items()}
    after = digest_parameters(model)
    lens_by_layer = []
    for layer_index in range(len(layers)):
        values = [row["logit_lens_gold_minus_demand_by_layer"][layer_index] for row in rows
                  if row["arm"] == "original" and row["gold"] != "Demand Letter"]
        lens_by_layer.append(sum(values) / len(values))
    summary = {
        "model_revision": MODEL_REVISION,
        "run": "form-type-decision-v1",
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "modified_parameters": 0,
        "first_label_tokens": {label: [token, tokenizer.decode([token])] for label, token in first_token.items()},
        "option_token_tails": {str(option): tail for option, tail in option_tails.items()},
        "arms": {arm: arm_summary(arm) for arm in ("original", "reversed_mapping")},
        "probe_nearest_centroid_loo_accuracy": probe,
        "logit_lens_mean_gold_minus_demand_by_layer": lens_by_layer,
        "attribution_mean_gold_minus_demand": dict(sorted(mean_attribution.items(), key=lambda item: item[1])),
        "attribution_cases": len(next(iter(attribution.values()))),
        "component_reconstruction_relative_error": reconstruction_error,
        "lens_vs_model_logit_max_abs_error": lens_logits_check,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({arm: {key: value for key, value in summary["arms"][arm].items()
                            if key in ("constrained_accuracy", "greedy_exact_label_accuracy", "constrained_choice_counts")}
                      for arm in summary["arms"]}, indent=2), flush=True)
    print("probe", json.dumps(probe), flush=True)
    print("hash_same", before == after, "reconstruction", reconstruction_error, "lens_check", lens_logits_check, flush=True)
    if before != after:
        raise AssertionError("parameters changed")


if __name__ == "__main__":
    main()

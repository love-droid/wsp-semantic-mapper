"""Frozen pointer test for layer-14 head 5 on form_type. Never edits weights.

heads-v1 found head 5 attends to the first canonical-label mapping key and writes it. Zeroing
the head stops the copy but does not make the gold label win. This tests whether head 5 is a
faithful pointer: if its attention on the first key is moved to the gold key, does the gold
label win? A wrong-key move is the control. Also records how head 5 naturally ranks the keys.
Development claim_form cases only. Eager attention throughout so weights and outputs agree.
"""
import json
import random
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt


LAYER = 14
HEAD = 5
HEAD_DIM = 64
GREEDY_MAX_NEW = 16
MAPPING_LABELS = ["Proof of Loss", "Employer's First Report", "Loss Run", "Demand Letter"]
RUN_DIR = ROOT / "experiments/form_type/pointer-v1"
RULES = {
    "question": "If layer-14 head 5 read the gold mapping key instead of the first key, would the gold label win?",
    "cases": "122 development claim_form.yaml cases, original and reversed_mapping arms. Held-out not loaded.",
    "key_token": "Token holding the first character of each label where it appears as a key in the escaped mapping dict.",
    "arms": {
        "base": "Unmodified eager forward.",
        "swap_gold": "Head-5 attention weight on the first-key token is moved onto the gold-key token. Output changes by a_first * (v_gold - v_first).",
        "swap_wrong": "Same move onto a non-first, non-gold key chosen with a fixed seed. Control.",
        "onehot_gold": "Head-5 output replaced by the gold-key value vector alone.",
        "onehot_wrong": "Head-5 output replaced by the wrong-key value vector alone. Control.",
    },
    "applies_to": "Label decision position only. Cases whose gold is the first key are excluded from swap arms.",
    "metrics": "First-token label accuracy among five labels; greedy exact label for swap_gold and onehot_gold.",
    "weights": "Frozen.",
}


def main():
    cases = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    cases = [case for case in cases if case["schema_name"] == "claim_form.yaml"]
    if len(cases) != 122:
        raise AssertionError(len(cases))
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    rng = random.Random(20261108)

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    print("parameter sha256", before, flush=True)
    layers = model.model.layers
    attention_index = sum(layer.is_attention_layer for layer in layers[:LAYER])
    attention = layers[LAYER].self_attn
    kv_head = HEAD // attention.num_key_value_groups

    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}

    records = []
    for arm in ("original", "reversed_mapping"):
        for case in cases:
            built = case if arm == "original" else reversed_case(case)
            mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
            messages, _ = build_prompt(built)
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            if not isinstance(ids, list):
                ids = ids.tolist()
            ids = ids + quote_ids
            pieces = [tokenizer.decode([token]) for token in ids]
            starts, cursor = [], 0
            for piece in pieces:
                starts.append(cursor)
                cursor += len(piece)
            joined = "".join(pieces)
            mark = joined.find(MAPPING_MARK.strip())
            key_positions = {}
            for label in MAPPING_LABELS:
                at = joined.find('\\"' + label + '\\": [', mark)
                if at < 0:
                    raise AssertionError(f"{label} key not found in {case['id']}")
                char = at + 2
                key_positions[label] = max(position for position, start in enumerate(starts) if start <= char)
            gold = case["expected"]["form_type"]
            first_key = next(iter(mapping))
            wrong_pool = [label for label in MAPPING_LABELS if label not in (gold, first_key)]
            records.append({"id": case["id"], "arm": arm, "gold": gold, "first_key": first_key,
                            "wrong": rng.choice(wrong_pool), "ids": ids, "key_positions": key_positions,
                            "key_tokens": {label: pieces[position] for label, position in key_positions.items()}})

    started = time.perf_counter()
    state = {}

    def capture_values(module, args, output):
        state["values"] = output.float()

    def replace_head(module, args):
        if "replacement" not in state or args[0].shape[1] == 1:
            return None
        value = args[0].clone()
        value[0, -1, HEAD * HEAD_DIM:(HEAD + 1) * HEAD_DIM] = state["replacement"].to(value.dtype)
        return (value,)

    def grab_head(module, args):
        state["head_out"] = args[0][0, -1, HEAD * HEAD_DIM:(HEAD + 1) * HEAD_DIM].float().clone()

    hooks = [attention.v_proj.register_forward_hook(capture_values),
             attention.out_proj.register_forward_pre_hook(grab_head),
             attention.out_proj.register_forward_pre_hook(replace_head)]

    def label_choice(logits):
        return max(LABELS, key=lambda label: float(logits[first_token[label]]))

    def greedy(ids):
        output = model.generate(torch.tensor([ids], device="cuda"), do_sample=False,
                                max_new_tokens=GREEDY_MAX_NEW, repetition_penalty=1.0, use_cache=True)
        return tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)

    rows = []
    with torch.inference_mode():
        for number, record in enumerate(records):
            state.pop("replacement", None)
            ids = torch.tensor([record["ids"]], device="cuda")
            out = model(input_ids=ids, output_attentions=True, use_cache=False, logits_to_keep=1)
            base_logits = out.logits[0, -1].float()
            weights = out.attentions[attention_index][0, HEAD, -1].float()
            values = state["values"][0, :, kv_head * HEAD_DIM:(kv_head + 1) * HEAD_DIM]
            natural = state["head_out"]
            reconstructed = weights @ values
            key_attention = {label: float(weights[position]) for label, position in record["key_positions"].items()}
            row = {
                "id": record["id"], "arm": record["arm"], "gold": record["gold"],
                "first_key": record["first_key"], "wrong": record["wrong"],
                "key_tokens": record["key_tokens"], "key_attention": key_attention,
                "head_output_reconstruction_error": float((reconstructed - natural).abs().max()),
                "base_choice": label_choice(base_logits),
                "top1_source_position_is_first_key": int(weights.argmax()) == record["key_positions"][record["first_key"]],
            }
            first_position = record["key_positions"][record["first_key"]]
            if record["gold"] != record["first_key"]:
                gold_position = record["key_positions"][record["gold"]]
                wrong_position = record["key_positions"][record["wrong"]]
                moved = weights[first_position]
                variants = {
                    "swap_gold": natural + moved * (values[gold_position] - values[first_position]),
                    "swap_wrong": natural + moved * (values[wrong_position] - values[first_position]),
                    "onehot_gold": values[gold_position],
                    "onehot_wrong": values[wrong_position],
                }
                for name, replacement in variants.items():
                    state["replacement"] = replacement
                    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    row[name] = label_choice(logits)
                    row[name + "_gold_logit_minus_first_key"] = float(
                        logits[first_token[record["gold"]]] - logits[first_token[record["first_key"]]])
                for name in ("swap_gold", "onehot_gold"):
                    state["replacement"] = variants[name]
                    row[name + "_greedy"] = greedy(record["ids"])
                state.pop("replacement", None)
            rows.append(row)
            if number % 40 == 0:
                print(f"{number}/{len(records)} {time.perf_counter() - started:.1f}s", flush=True)
    for hook in hooks:
        hook.remove()
    after = digest_parameters(model)
    with (RUN_DIR / "records.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    def summarize(arm):
        chosen = [row for row in rows if row["arm"] == arm]
        moved = [row for row in chosen if "swap_gold" in row]
        result = {
            "cases": len(chosen), "swap_cases": len(moved),
            "base_label_accuracy_on_swap_cases": sum(row["base_choice"] == row["gold"] for row in moved) / len(moved),
            "head5_top1_is_first_key": sum(row["top1_source_position_is_first_key"] for row in chosen) / len(chosen),
            "mean_key_attention": {label: sum(row["key_attention"][label] for row in chosen) / len(chosen) for label in MAPPING_LABELS},
            "gold_rank_among_non_first_keys": Counter(
                1 + sorted([label for label in MAPPING_LABELS if label != row["first_key"]],
                           key=lambda label: -row["key_attention"][label]).index(row["gold"])
                for row in moved),
        }
        for name in ("swap_gold", "swap_wrong", "onehot_gold", "onehot_wrong"):
            target = "gold" if name.endswith("gold") else "wrong"
            result[name] = {
                "target_label_chosen": sum(row[name] == row[target] for row in moved) / len(moved),
                "gold_chosen": sum(row[name] == row["gold"] for row in moved) / len(moved),
                "first_key_chosen": sum(row[name] == row["first_key"] for row in moved) / len(moved),
                "median_gold_minus_first_key": sorted(row[name + "_gold_logit_minus_first_key"] for row in moved)[len(moved) // 2],
            }
        for name in ("swap_gold", "onehot_gold"):
            result[name + "_greedy_exact"] = sum(row[name + "_greedy"].split('"')[0] == row["gold"] for row in moved) / len(moved)
            result[name + "_greedy_labels"] = Counter(row[name + "_greedy"].split('"')[0] for row in moved).most_common(8)
        return result

    summary = {
        "model_revision": MODEL_REVISION, "run": "form-type-pointer-v1",
        "layer": LAYER, "head": HEAD, "kv_head": kv_head,
        "parameter_sha256_before": before, "parameter_sha256_after": after, "modified_parameters": 0,
        "max_head_output_reconstruction_error": max(row["head_output_reconstruction_error"] for row in rows),
        "key_token_examples": Counter(json.dumps(row["key_tokens"]) for row in rows).most_common(2),
        "arms": {arm: summarize(arm) for arm in ("original", "reversed_mapping")},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    if before != after:
        raise AssertionError("parameters changed")


if __name__ == "__main__":
    main()

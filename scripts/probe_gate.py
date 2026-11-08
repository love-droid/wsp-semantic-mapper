"""Frozen check for a context gate at the form_type decision. Never edits weights. Fit half only.

Candidates 2-4 pushed the label wherever document-type units were active, and changed how outputs
begin. A repurposed MLP unit could act only at the decision if some direction in the FFN input
separates the decision position from every other prompt and generated position. This measures
that separation with a held-back half of positions, the bias-like constant channels a gate
without a bias term would need, class readout at the decision, and which units are close to idle.
"""
import hashlib
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_form_type import PREFIX, reversed_case
from run_candidate1 import split_cases
from run_candidate4 import GENERIC_PROMPTS, GENERIC_SYSTEM, ORIGINAL_OUTPUTS
from run_controls import generate_all
from run_fieldbench import DATA_DIR, ROOT, build_prompt


LAYERS = (14, 15)
PROMPT_SAMPLES = 64
RUN_DIR = ROOT / "experiments/form_type/gate-v1"


def main():
    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    fit, _ = split_cases([case for case in dev if case["schema_name"] == "claim_form.yaml"])
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261108)
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)

    def chat_ids(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        return ids if isinstance(ids, list) else ids.tolist()

    decisions = []
    for arm in ("original", "reversed_mapping"):
        for case in fit:
            built = case if arm == "original" else reversed_case(case)
            messages, _ = build_prompt(built)
            decisions.append({"id": case["id"], "gold": case["expected"]["form_type"], "ids": chat_ids(messages) + quote_ids})
    outputs = {row["id"]: row["raw_output"] for row in map(json.loads, ORIGINAL_OUTPUTS.open())}
    sequences = []
    for case in fit:
        messages, _ = build_prompt(case)
        prompt = chat_ids(messages)
        sequences.append({"id": case["id"], "prompt_len": len(prompt),
                          "ids": prompt + tokenizer.encode(outputs[case["id"]], add_special_tokens=False)})
    generic = [chat_ids([{"role": "system", "content": GENERIC_SYSTEM}, {"role": "user", "content": text}]) for text in GENERIC_PROMPTS]
    with torch.inference_mode():
        answers = generate_all(model, tokenizer, generic, 64)
    for index, (prompt, answer) in enumerate(zip(generic, answers)):
        sequences.append({"id": f"generic-{index}", "prompt_len": len(prompt), "ids": prompt + tokenizer.encode(answer, add_special_tokens=False)})

    captured = {}
    hooks = []
    for layer_index in LAYERS:
        ffn = model.model.layers[layer_index].feed_forward
        hooks.append(ffn.register_forward_pre_hook(lambda m, a, i=layer_index: captured.__setitem__(f"x{i}", a[0][0].float())))
        hooks.append(ffn.w2.register_forward_pre_hook(lambda m, a, i=layer_index: captured.__setitem__(f"a{i}", a[0][0].float())))
    data = {layer_index: {"dec_x": [], "dec_gold": [], "dec_half": [], "prot_x": [], "prot_half": [], "prot_kind": [], "prot_a": [], "dec_a": []}
            for layer_index in LAYERS}
    with torch.inference_mode():
        for record in decisions:
            model(input_ids=torch.tensor([record["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            half = hash_half(record["id"])
            samples = rng.sample(list(range(0, len(record["ids"]) - 8)), PROMPT_SAMPLES)
            for layer_index in LAYERS:
                x, a = captured[f"x{layer_index}"], captured[f"a{layer_index}"]
                bucket = data[layer_index]
                bucket["dec_x"].append(x[-1]); bucket["dec_a"].append(a[-1])
                bucket["dec_gold"].append(record["gold"]); bucket["dec_half"].append(half)
                for position in samples:
                    bucket["prot_x"].append(x[position]); bucket["prot_a"].append(a[position])
                    bucket["prot_half"].append(half); bucket["prot_kind"].append("prompt")
        for sequence in sequences:
            model(input_ids=torch.tensor([sequence["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            half = hash_half(sequence["id"])
            kind = "generic" if sequence["id"].startswith("generic") else "claim_output"
            line, positions = "", []
            for offset, token in enumerate(sequence["ids"][sequence["prompt_len"]:]):
                line = (line + tokenizer.decode([token])).rsplit("\n", 1)[-1]
                if "form_type" not in line:
                    positions.append(sequence["prompt_len"] + offset - 1)
            for layer_index in LAYERS:
                x, a = captured[f"x{layer_index}"], captured[f"a{layer_index}"]
                bucket = data[layer_index]
                for position in positions:
                    bucket["prot_x"].append(x[position]); bucket["prot_a"].append(a[position])
                    bucket["prot_half"].append(half); bucket["prot_kind"].append(kind)
    for hook in hooks:
        hook.remove()

    summary = {"parameter_sha256": before}
    for layer_index in LAYERS:
        bucket = data[layer_index]
        dec_x = torch.stack(bucket["dec_x"]).double()
        prot_x = torch.stack(bucket["prot_x"]).double()
        dec_half = torch.tensor(bucket["dec_half"], device="cuda")
        prot_half = torch.tensor(bucket["prot_half"], device="cuda")
        result = {"decisions": len(dec_x), "protected": len(prot_x)}
        for train, evaluate in ((0, 1), (1, 0)):
            d_train, p_train = dec_x[dec_half == train], prot_x[prot_half == train]
            d_eval, p_eval = dec_x[dec_half == evaluate], prot_x[prot_half == evaluate]
            covariance = torch.cov(p_train.T) + 1e-3 * torch.eye(dec_x.shape[1], dtype=torch.float64, device="cuda")
            direction = torch.linalg.solve(covariance, d_train.mean(0) - p_train.mean(0))
            d_score, p_score = d_eval @ direction, p_eval @ direction
            threshold = float(d_score.min())
            kinds = [kind for kind, half in zip(bucket["prot_kind"], bucket["prot_half"]) if half == evaluate]
            above = (p_score >= threshold)
            result[f"train{train}_eval{evaluate}"] = {
                "decision_min": threshold, "decision_median": float(d_score.median()),
                "protected_max": float(p_score.max()), "protected_p999": float(torch.quantile(p_score.float(), 0.999)),
                "protected_at_or_above_decision_min": int(above.sum()), "protected_eval": len(p_score),
                "above_by_kind": {kind: int(sum(flag for flag, k in zip(above.tolist(), kinds) if k == kind)) for kind in set(kinds)},
                "margin_std_units": float((d_score.min() - p_score.max()) / p_score.std()),
            }
        all_x = torch.cat([dec_x, prot_x])
        ratio = (all_x.mean(0).abs() / (all_x.std(0) + 1e-6))
        top = torch.topk(ratio, 5)
        result["bias_like_channels"] = [{"dim": int(i), "mean": float(all_x[:, i].mean()), "std": float(all_x[:, i].std()),
                                         "mean_over_std": float(r)} for r, i in zip(top.values, top.indices)]
        golds = bucket["dec_gold"]
        classes = sorted(set(golds))
        standardized = (dec_x - dec_x.mean(0)) / (dec_x.std(0) + 1e-6)
        correct = 0
        for held in range(len(golds)):
            centroids = {c: standardized[[i for i, g in enumerate(golds) if g == c and i != held]].mean(0) for c in classes}
            correct += min(classes, key=lambda c: float((standardized[held] - centroids[c]).norm())) == golds[held]
        result["decision_class_readout_loo"] = correct / len(golds)
        prot_a = torch.stack(bucket["prot_a"]).double()
        w2 = model.model.layers[layer_index].feed_forward.w2.weight.double()
        importance = prot_a.abs().mean(0) * w2.norm(dim=0)
        quiet = torch.topk(-importance, 12).indices
        result["quietest_units"] = [{"unit": int(u), "mean_abs_contribution": float(importance[u]),
                                     "max_abs_activation": float(prot_a[:, u].abs().max())} for u in quiet]
        result["median_unit_contribution"] = float(importance.median())
        summary[f"layer{layer_index}"] = result
        print(layer_index, json.dumps({k: v for k, v in result.items() if k.startswith("train") or k == "decision_class_readout_loo"}), flush=True)
    summary["parameter_sha256_after"] = digest_parameters(model)
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["parameter_sha256_after"] != before:
        raise AssertionError("parameters changed")


def hash_half(identifier):
    return int(hashlib.sha256(identifier.encode()).hexdigest(), 16) % 2


if __name__ == "__main__":
    main()

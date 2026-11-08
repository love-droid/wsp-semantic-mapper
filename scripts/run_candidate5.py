"""Candidate 5: four gated layer-15 MLP units that fire only at the form_type label decision.

gate-v1 found a direction in the layer-15 FFN input that separates the decision token from every
held-back prompt and generated position by 8-11 standard deviations. Four of the quietest units are
rewritten: w1 row = that gate (no bias, so silu is near zero elsewhere), w3 row = one-vs-rest
document-type readout, w2 column = push toward that label's first token. No new parameters, no
gradients; every fit is ridge least squares on fit-half data. GPU memory only.
"""
import hashlib
import json
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from candidate_eval import decision_choices, evaluate_all, pass_checks
from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from run_candidate1 import split_cases
from run_candidate4 import GENERIC_PROMPTS, GENERIC_SYSTEM, ORIGINAL_OUTPUTS
from run_controls import generate_all
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt


LAYER = 15
UNITS = [34, 235, 4572, 2265]
MAPPING_LABELS = ["Demand Letter", "Employer's First Report", "Loss Run", "Proof of Loss"]
PROMPT_SAMPLES = 64
GATE_ON = 6.0
GATE_OFF = -2.0
MAX_GATE_LEAK = 0.001
GATE_REWEIGHT_ROUNDS = 5
GAINS = (0.0, 8.0, 12.0, 16.0, 24.0)
MAX_FLIP = 0.001
STRUCTURE_TOKENS = 20
RUN_DIR = ROOT / "experiments/candidates/candidate5-l15-gated-units-r3"
RULES = {
    "revision_r3": "r2 passed the gate check and aborted at selection: the gain-0 ablation itself flips 2 of the first-20 generated tokens, so the zero-structural-flip rule could not be met by any edit of these units. r3 compares structural flips with the gain-0 ablation. Fit-half only; test half untouched.",
    "revision": "r1 aborted at the gate check before any evaluation: one held-back half had protected max +0.07 against decision min +0.85. r2 reweights near-threshold protected positions (5 rounds, x10) and scales so the lowest held-back decision is +6; abort if more than 0.1% of held-back protected positions exceed -2.",
    "tensors": "model.layers.15.feed_forward.{w1,w3}.weight rows [34, 235, 4572, 2265] and w2.weight columns [34, 235, 4572, 2265]; 3 x 4 x 1024 = 12,288 entries",
    "units": "Four lowest mean |activation| x ||w2 column|| over protected positions in gate-v1 (fit half).",
    "gate": "Homogeneous ridge least squares on FFN input: +1 at decision tokens, -1 at protected positions, class-balanced, protected positions above -0.5 reweighted x10 for 5 rounds. Scaled so the lowest decision is +6. Abort if more than 0.1% of held-back protected positions exceed -2.",
    "readout": "w3 row c: ridge least squares to 1 at decisions of class c, 0 at other decisions and at protected positions.",
    "push": "w2 column c: gain logits toward label c's first token over the other labels, divided by the mean decision gate activation.",
    "grid": {"gain": GAINS, "note": "gain 0 is the ablation of the four original units; never selected."},
    "selection": "Highest fit-half first-token label accuracy with greedy-token change <= 0.1% at protected generated positions and no more changes in the first 20 generated tokens than the gain-0 ablation. Ties take the smaller gain.",
    "pass": "Same as candidates 1-4; drops compared after rounding to 1e-9.",
    "held_out": "Not loaded here.",
    "weights_on_disk": "Unchanged. edit.pt written for the held-out step.",
}


def hash_half(identifier):
    return int(hashlib.sha256(identifier.encode()).hexdigest(), 16) % 2


def main():
    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    claims = [case for case in dev if case["schema_name"] == "claim_form.yaml"]
    if len(dev) != 200 or len(claims) != 122:
        raise AssertionError((len(dev), len(claims)))
    fit, test = split_cases(claims)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    rng = random.Random(20261108)

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    original_hash = digest_parameters(model)
    print("parameter sha256", original_hash, flush=True)
    ffn = model.model.layers[LAYER].feed_forward
    originals = {name: getattr(ffn, name).weight.data.clone() for name in ("w1", "w3", "w2")}
    units = torch.tensor(UNITS, device="cuda")
    gamma = model.model.embedding_norm.weight.float()
    unembed = model.lm_head.weight.float()
    eps = model.model.embedding_norm.variance_epsilon
    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}

    def chat_ids(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        return ids if isinstance(ids, list) else ids.tolist()

    def decision_record(case, arm, split_name):
        built = case if arm == "original" else reversed_case(case)
        messages, _ = build_prompt(built)
        mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
        return {"id": case["id"], "split": split_name, "arm": arm, "gold": case["expected"]["form_type"],
                "first_key": next(iter(mapping)), "ids": chat_ids(messages) + quote_ids}

    fit_records = [decision_record(case, arm, "fit") for arm in ("original", "reversed_mapping") for case in fit]
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
    for sequence in sequences:
        line, positions = "", []
        for offset, token in enumerate(sequence["ids"][sequence["prompt_len"]:]):
            line = (line + tokenizer.decode([token])).rsplit("\n", 1)[-1]
            if "form_type" not in line:
                positions.append(sequence["prompt_len"] + offset - 1)
        sequence["positions"] = positions
        sequence["structure"] = [p for p in positions if p < sequence["prompt_len"] - 1 + STRUCTURE_TOKENS]

    started = time.perf_counter()
    captured = {}
    hooks = [ffn.register_forward_pre_hook(lambda m, a: captured.__setitem__("x", a[0][0].float())),
             model.model.layers[LAYER].register_forward_hook(
                 lambda m, a, o: captured.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))]
    dec_x, dec_gold, dec_half, rms_inv = [], [], [], []
    prot_x, prot_half = [], []
    base_argmax = []
    with torch.inference_mode():
        for record in fit_records:
            model(input_ids=torch.tensor([record["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
            x = captured["x"]
            dec_x.append(x[-1]); dec_gold.append(record["gold"]); dec_half.append(hash_half(record["id"]))
            rms_inv.append(float(torch.rsqrt(captured["r"].pow(2).mean() + eps)))
            for position in rng.sample(list(range(0, x.shape[0] - 8)), PROMPT_SAMPLES):
                prot_x.append(x[position]); prot_half.append(hash_half(record["id"]))
        for sequence in sequences:
            logits = model(input_ids=torch.tensor([sequence["ids"]], device="cuda"), use_cache=False, logits_to_keep=0).logits[0]
            base_argmax.append(logits[sequence["positions"]].argmax(-1))
            x = captured["x"]
            for position in sequence["positions"]:
                prot_x.append(x[position]); prot_half.append(hash_half(sequence["id"]))
    for hook in hooks:
        hook.remove()
    dec_x = torch.stack(dec_x).double()
    prot_x = torch.stack(prot_x).double()
    dec_half = torch.tensor(dec_half, device="cuda")
    prot_half = torch.tensor(prot_half, device="cuda")
    dim = dec_x.shape[1]
    print(f"capture {time.perf_counter() - started:.1f}s decisions {len(dec_x)} protected {len(prot_x)}", flush=True)

    def fit_gate(d, p):
        weight_d = torch.full((len(d),), 1.0 / len(d), dtype=torch.float64, device="cuda")
        weight_p = torch.full((len(p),), 1.0 / len(p), dtype=torch.float64, device="cuda")
        for _ in range(GATE_REWEIGHT_ROUNDS):
            gram = (d * weight_d[:, None]).T @ d + (p * weight_p[:, None]).T @ p
            rhs = (d * weight_d[:, None]).sum(0) - (p * weight_p[:, None]).sum(0)
            ridge = 1e-3 * torch.trace(gram) / dim
            direction = torch.linalg.solve(gram + ridge * torch.eye(dim, dtype=torch.float64, device="cuda"), rhs)
            weight_p = torch.where(p @ direction > -0.5, weight_p * 10, weight_p)
        return direction

    validation = []
    for train, held in ((0, 1), (1, 0)):
        direction = fit_gate(dec_x[dec_half == train], prot_x[prot_half == train])
        d_score, p_score = dec_x[dec_half == held] @ direction, prot_x[prot_half == held] @ direction
        scale = GATE_ON / float(d_score.min()) if float(d_score.min()) > 0 else None
        entry = {"train_half": train, "protected_max_raw": float(p_score.max()), "decision_min_raw": float(d_score.min())}
        if scale:
            entry["protected_max_scaled"] = float(p_score.max() * scale)
            entry["protected_above_off_fraction"] = float((p_score * scale > GATE_OFF).double().mean())
        validation.append(entry)
    print("gate validation", json.dumps(validation), flush=True)
    if any(entry.get("protected_above_off_fraction", 1.0) > MAX_GATE_LEAK for entry in validation):
        (RUN_DIR / "verdict.json").write_text(json.dumps({"aborted": "gate not clean on held-back fit positions", "validation": validation}, indent=2) + "\n")
        raise SystemExit("aborted: gate not clean")
    direction = fit_gate(dec_x, prot_x)
    scale = GATE_ON / float((dec_x @ direction).min())
    gate_row = direction * scale
    dec_gate = dec_x @ gate_row
    print(f"gate on decisions min {float(dec_gate.min()):.2f} median {float(dec_gate.median()):.2f}; protected max {float((prot_x @ gate_row).max()):.2f}", flush=True)

    readouts = []
    for label in MAPPING_LABELS:
        features = torch.cat([dec_x, prot_x])
        target = torch.cat([torch.tensor([1.0 if g == label else 0.0 for g in dec_gold], dtype=torch.float64, device="cuda"),
                            torch.zeros(len(prot_x), dtype=torch.float64, device="cuda")])
        weights = torch.cat([torch.full((len(dec_x),), 1.0 / len(dec_x), dtype=torch.float64, device="cuda"),
                             torch.full((len(prot_x),), 1.0 / len(prot_x), dtype=torch.float64, device="cuda")])
        gram = (features * weights[:, None]).T @ features
        ridge = 1e-3 * torch.trace(gram) / dim
        readouts.append(torch.linalg.solve(gram + ridge * torch.eye(dim, dtype=torch.float64, device="cuda"), (features * weights[:, None]).T @ target))
    readouts = torch.stack(readouts)
    readout_values = dec_x @ readouts.T
    readout_accuracy = sum(MAPPING_LABELS[int(row.argmax())] == gold for row, gold in zip(readout_values, dec_gold)) / len(dec_gold)
    activation = F.silu(dec_gate)
    mean_activation = float(activation.mean())
    mean_rms = sum(1.0 / value for value in rms_inv) / len(rms_inv)
    pushes = []
    for label in MAPPING_LABELS:
        others = torch.stack([unembed[first_token[o]] for o in LABELS if o != label]).mean(0)
        e = gamma * (unembed[first_token[label]] - others)
        pushes.append((e / e.pow(2).sum()).double() * mean_rms / mean_activation)
    pushes = torch.stack(pushes, dim=1)
    print(f"fit readout accuracy {readout_accuracy:.3f}; mean gate activation {mean_activation:.2f}", flush=True)

    def apply(gain):
        ffn.w1.weight.data[units] = gate_row.unsqueeze(0).expand(len(UNITS), -1).to(ffn.w1.weight.dtype)
        ffn.w3.weight.data[units] = readouts.to(ffn.w3.weight.dtype)
        ffn.w2.weight.data[:, units] = (gain * pushes).to(ffn.w2.weight.dtype)

    def restore():
        for name, value in originals.items():
            getattr(ffn, name).weight.data.copy_(value)

    def flip_counts():
        flips = structural = 0
        total = 0
        with torch.inference_mode():
            for sequence, base in zip(sequences, base_argmax):
                logits = model(input_ids=torch.tensor([sequence["ids"]], device="cuda"), use_cache=False, logits_to_keep=0).logits[0]
                changed = logits[sequence["positions"]].argmax(-1) != base
                flips += int(changed.sum())
                total += len(sequence["positions"])
                structural += int(changed[:len(sequence["structure"])].sum())
        return flips / total, structural

    grid = []
    for gain in GAINS:
        apply(gain)
        choices = decision_choices(model, tokenizer, fit_records, first_token)
        flip_rate, structural = flip_counts()
        restore()
        entry = {"gain": gain, "fit_label_accuracy": sum(c == r["gold"] for c, r in zip(choices, fit_records)) / len(fit_records),
                 "generated_flip_rate": flip_rate, "structural_flips": structural}
        entry["eligible"] = gain > 0 and flip_rate <= MAX_FLIP and structural <= grid[0]["structural_flips"]
        grid.append(entry)
        print("grid", json.dumps(entry), flush=True)
    if digest_parameters(model) != original_hash:
        raise AssertionError("grid restore failed")
    eligible = [entry for entry in grid if entry["eligible"]]
    if not eligible:
        (RUN_DIR / "verdict.json").write_text(json.dumps({"aborted": "no gain within limits", "grid": grid, "validation": validation}, indent=2) + "\n")
        raise SystemExit("aborted: no gain within limits")
    chosen = max(eligible, key=lambda entry: (entry["fit_label_accuracy"], -entry["gain"]))
    print("chosen", json.dumps(chosen), flush=True)

    decision_records = [decision_record(case, arm, split_name)
                        for split_name, group in (("fit", fit), ("test", test))
                        for arm in ("original", "reversed_mapping") for case in group]
    original = evaluate_all(model, tokenizer, "original", RUN_DIR, dev, fit, test, decision_records, first_token)
    if digest_parameters(model) != original_hash:
        raise AssertionError("original evaluation changed parameters")
    apply(chosen["gain"])
    edited_hash = digest_parameters(model)
    changed = {}
    for name, value in originals.items():
        diff = getattr(ffn, name).weight.data != value
        outside = diff.clone()
        if name == "w2":
            outside[:, units] = False
        else:
            outside[units] = False
        if bool(outside.any()):
            raise AssertionError(f"{name} changed outside the four units")
        changed[name] = int(diff.sum())
    edit_info = {"tensors": [f"model.layers.{LAYER}.feed_forward.{name}.weight" for name in ("w1", "w3", "w2")],
                 "units": UNITS, "entries": 3 * len(UNITS) * dim, "entries_changed_after_bf16": changed,
                 "parameter_sha256_edited": edited_hash, "readout_fit_accuracy": readout_accuracy,
                 "gate_validation": validation, "mean_gate_activation": mean_activation}
    torch.save({"slices": [
        {"tensor": f"model.layers.{LAYER}.feed_forward.w1.weight", "dim": 0, "index": units.cpu(), "values": ffn.w1.weight.data[units].cpu().clone()},
        {"tensor": f"model.layers.{LAYER}.feed_forward.w3.weight", "dim": 0, "index": units.cpu(), "values": ffn.w3.weight.data[units].cpu().clone()},
        {"tensor": f"model.layers.{LAYER}.feed_forward.w2.weight", "dim": 1, "index": units.cpu(), "values": ffn.w2.weight.data[:, units].cpu().clone()},
    ]}, RUN_DIR / "edit.pt")
    print("edit", json.dumps(edit_info), flush=True)
    edited = evaluate_all(model, tokenizer, "edited", RUN_DIR, dev, fit, test, decision_records, first_token)
    restore()
    restored_hash = digest_parameters(model)
    checks = pass_checks(original, edited)
    verdict = {
        "model_revision": MODEL_REVISION, "candidate": RUN_DIR.name,
        "parameter_sha256_original": original_hash, "parameter_sha256_restored": restored_hash,
        "grid": grid, "chosen": chosen, "edit": edit_info, "original": original, "edited": edited, "checks": checks,
        "elapsed_seconds": time.perf_counter() - started, "held_out_used": False, "weights_on_disk_modified": False,
    }
    (RUN_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    print("checks", json.dumps(checks, indent=1), flush=True)
    if restored_hash != original_hash:
        raise AssertionError("restore failed")


if __name__ == "__main__":
    main()

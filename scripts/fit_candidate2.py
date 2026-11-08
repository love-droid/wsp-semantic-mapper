"""Fit-half dry run for candidate 2: sparse layer-15 w2 column edit. Never evaluates the test half.

Reads only the 61 fit claim cases from run_candidate1's split. Chooses 64 w2 input units that
separate the gold form type at the label decision token and are quiet elsewhere, solves a ridge
least-squares edit of those 64 columns that adds a gold-label push to the final residual, and
reports size and fit-half decision accuracy for a small grid. The edit lives in GPU memory and is
reverted before exit.
"""
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from run_candidate1 import split_cases
from run_fieldbench import DATA_DIR, ROOT, build_prompt


LAYER = 15
UNITS = 64
BATCH = 8
PROTECT_PER_PROMPT = 64
GAINS = (4.0, 8.0, 12.0)
MUS = (1.0, 10.0)
RUN_DIR = ROOT / "experiments/candidates/candidate2-dryrun-fit-only"


def main():
    dev = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    claims = [case for case in dev if case["schema_name"] == "claim_form.yaml"]
    fit, _ = split_cases(claims)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261108)
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    original_hash = digest_parameters(model)
    w2 = model.model.layers[LAYER].feed_forward.w2
    original_w2 = w2.weight.data.clone()
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
            mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
            records.append({"gold": case["expected"]["form_type"], "first_key": next(iter(mapping)),
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
    h_k = hidden[:, units]
    p_k = protect[:, units]

    directions = {}
    for label in LABELS:
        others = torch.stack([unembed[first_token[o]] for o in LABELS if o != label]).mean(0)
        e = gamma * (unembed[first_token[label]] - others)
        directions[label] = (e / e.pow(2).sum()).double()

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

    def accuracy():
        correct = first = 0
        with torch.inference_mode():
            for start in range(0, len(records), BATCH):
                chunk = records[start:start + BATCH]
                encoded, positions = pad([r["ids"] for r in chunk])
                logits = model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                for row, record in zip(logits, chunk):
                    choice = max(LABELS, key=lambda label: float(row[first_token[label]]))
                    correct += choice == record["gold"]
                    first += choice == record["first_key"]
        return correct / len(records), first / len(records)

    base_accuracy, base_first = accuracy()
    columns = original_w2[:, units].double()
    full_protect_out = (protect @ original_w2.double().T).norm()
    grid = []
    for gain in GAINS:
        for mu in MUS:
            delta = solve(gain, mu)
            w2.weight.data[:, units] = (columns + delta).to(w2.weight.dtype)
            acc, first = accuracy()
            w2.weight.data.copy_(original_w2)
            grid.append({"gain": gain, "mu": mu, "fit_label_accuracy": acc, "fit_first_key_choice": first,
                         "relative_frobenius_vs_columns": float(delta.norm() / columns.norm()),
                         "relative_frobenius_vs_tensor": float(delta.norm() / original_w2.double().norm()),
                         "protected_output_relative_change": float((p_k @ delta.T).norm() / full_protect_out)})
            print(json.dumps(grid[-1]), flush=True)
    if digest_parameters(model) != original_hash:
        raise AssertionError("restore failed")
    summary = {"layer": LAYER, "tensor": f"model.layers.{LAYER}.feed_forward.w2.weight", "shape": list(original_w2.shape),
               "units": units.tolist(), "entries": int(original_w2.shape[0] * UNITS),
               "fit_prompts": len(records), "base_fit_label_accuracy": base_accuracy, "base_fit_first_key_choice": base_first,
               "grid": grid, "test_half_used": False, "parameter_sha256": original_hash}
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("base", base_accuracy, base_first, flush=True)


if __name__ == "__main__":
    main()

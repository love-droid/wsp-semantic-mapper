"""Candidate 4: layer-15 w2 column edit protected at generated positions.

Candidates 2 and 3 fixed form_type but changed about half of all development outputs, because the
chosen units fire while the model writes. Here units are chosen to be quiet at the model's own
generated positions (fit-half claim outputs and 40 generic prompts that are not in the control
suite), the ridge penalty includes those positions, and a setting is rejected if it changes the
greedy token at more than 1% of them. Fit-half data only for every choice. GPU memory only.
"""
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from candidate_eval import decision_choices, evaluate_all, pass_checks
from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from run_candidate1 import split_cases
from run_controls import generate_all
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt


LAYER = 15
UNITS = 64
PROMPT_SAMPLES = 64
GENERATED_SAMPLES = 128
GAINS = (8.0, 12.0, 16.0)
MUS = (10.0, 30.0, 100.0)
MAX_TENSOR_FRACTION = 0.10
MAX_GENERATED_FLIP = 0.01
RUN_DIR = ROOT / "experiments/candidates/candidate4-l15w2-generation-protected"
ORIGINAL_OUTPUTS = ROOT / "experiments/candidates/candidate2-l15w2-columns/original_fieldbench_predictions.jsonl"
GENERIC_SYSTEM = "You are a helpful assistant. Answer briefly."
GENERIC_PROMPTS = [
    "What is the capital of France?", "Name three primary colors.", "Give a synonym for happy.",
    "What is the opposite of cold?", "Translate 'good morning' into Spanish.", "How many days are in a week?",
    "Write one sentence about the ocean.", "What gas do plants absorb?", "Name a mammal that can fly.",
    "Spell the word 'river' backwards.", "What is 7 times 6?", "List two fruits that are red.",
    "What color is the sky on a clear day?", "Give a word that rhymes with cat.", "What is the past tense of run?",
    "Name the largest planet.", "Write a short greeting for a colleague.", "What is frozen water called?",
    "Name a programming language.", "How many legs does a spider have?", "What is the boiling point of water in Celsius?",
    "Suggest a name for a pet fish.", "What do bees make?", "Give an example of a vegetable.",
    "Name a month with 30 days.", "What is the square root of 81?", "Write a JSON object with a key 'a' set to 1.",
    "What is the chemical symbol for gold?", "Describe a cat in five words.", "Name a musical instrument with strings.",
    "What shape has three sides?", "Give a word meaning very large.", "What season comes after winter?",
    "Name a country in South America.", "What is the plural of mouse?", "Write a Python list of three numbers.",
    "What is the freezing point of water in Fahrenheit?", "Name something you wear on your feet.",
    "What does CPU stand for?", "Give a polite way to decline an invitation.",
]
RULES = {
    "tensor": "model.layers.15.feed_forward.w2.weight, 64 columns, 1024 x 64 = 65,536 entries",
    "protection": "Prompt positions (64 per fit decision prompt) plus up to 128 generated positions per sequence from the original model's own outputs on the 61 fit claim cases and 40 generic prompts. The form_type line is excluded.",
    "units": "Top 64 by between-class variance at the decision token over (within-class variance + mean square at all protected positions).",
    "grid": {"gain": GAINS, "mu": MUS},
    "selection": "Highest fit-half first-token label accuracy among settings with relative Frobenius vs tensor <= 0.10 and greedy-token change at protected generated positions <= 1%. Ties take the smaller change. Test half not used.",
    "abort": "No setting within the limits.",
    "pass": "Same as candidates 1-3.",
    "held_out": "Not loaded here.",
    "weights_on_disk": "Unchanged. Passing edits are saved to edit.pt for the held-out step.",
}


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
    w2 = model.model.layers[LAYER].feed_forward.w2
    original_w2 = w2.weight.data.clone()
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
        sequences.append({"prompt_len": len(prompt), "ids": prompt + tokenizer.encode(outputs[case["id"]], add_special_tokens=False)})
    generic_prompts = [chat_ids([{"role": "system", "content": GENERIC_SYSTEM}, {"role": "user", "content": text}])
                       for text in GENERIC_PROMPTS]
    with torch.inference_mode():
        answers = generate_all(model, tokenizer, generic_prompts, 64)
    for prompt, answer in zip(generic_prompts, answers):
        sequences.append({"prompt_len": len(prompt), "ids": prompt + tokenizer.encode(answer, add_special_tokens=False)})
    for sequence in sequences:
        keep = []
        line = ""
        for offset, token in enumerate(sequence["ids"][sequence["prompt_len"]:]):
            piece = tokenizer.decode([token])
            line = (line + piece).rsplit("\n", 1)[-1]
            if "form_type" not in line:
                keep.append(sequence["prompt_len"] + offset)
        sequence["predict_positions"] = [position - 1 for position in keep if position - 1 >= sequence["prompt_len"] - 1]
        sequence["targets"] = keep

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
            for position in rng.sample(list(range(0, h.shape[0] - 8)), PROMPT_SAMPLES):
                protect.append(h[position])
            rms_inv.append(float(torch.rsqrt(captured["r"].pow(2).mean() + eps)))
            golds.append(record["gold"])
        base_argmax = []
        for sequence in sequences:
            ids = torch.tensor([sequence["ids"]], device="cuda")
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=0).logits[0]
            positions = sequence["predict_positions"]
            base_argmax.append(logits[positions].argmax(-1))
            h = captured["h"]
            for position in rng.sample(positions, min(GENERATED_SAMPLES, len(positions))):
                protect.append(h[position])
    for hook in hooks:
        hook.remove()
    hidden = torch.stack(hidden).double()
    protect = torch.stack(protect).double()
    total_generated = sum(len(sequence["predict_positions"]) for sequence in sequences)
    print(f"capture {time.perf_counter() - started:.1f}s protect {len(protect)} generated positions {total_generated}", flush=True)

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

    def solve(gain, mu):
        targets = torch.stack([gain / rms_inv[i] * directions[golds[i]] for i in range(len(golds))])
        second = p_k.T @ p_k / len(p_k) * len(h_k)
        system = h_k.T @ h_k + mu * second + 1e-6 * torch.trace(second) / UNITS * torch.eye(UNITS, dtype=torch.float64, device="cuda")
        return torch.linalg.solve(system, (targets.T @ h_k).T).T

    def generated_flip_rate():
        flips = 0
        with torch.inference_mode():
            for sequence, base in zip(sequences, base_argmax):
                logits = model(input_ids=torch.tensor([sequence["ids"]], device="cuda"), use_cache=False, logits_to_keep=0).logits[0]
                flips += int((logits[sequence["predict_positions"]].argmax(-1) != base).sum())
        return flips / total_generated

    columns = original_w2[:, units].double()
    tensor_norm = original_w2.double().norm()
    grid = []
    for gain in GAINS:
        for mu in MUS:
            delta = solve(gain, mu)
            entry = {"gain": gain, "mu": mu, "relative_frobenius_vs_tensor": float(delta.norm() / tensor_norm),
                     "relative_frobenius_vs_columns": float(delta.norm() / columns.norm())}
            if entry["relative_frobenius_vs_tensor"] <= MAX_TENSOR_FRACTION:
                w2.weight.data[:, units] = (columns + delta).to(w2.weight.dtype)
                choices = decision_choices(model, tokenizer, fit_records, first_token)
                entry["fit_label_accuracy"] = sum(c == r["gold"] for c, r in zip(choices, fit_records)) / len(fit_records)
                entry["generated_flip_rate"] = generated_flip_rate()
                w2.weight.data.copy_(original_w2)
            entry["eligible"] = entry.get("generated_flip_rate", 1.0) <= MAX_GENERATED_FLIP
            grid.append(entry)
            print("grid", json.dumps(entry), flush=True)
    if digest_parameters(model) != original_hash:
        raise AssertionError("grid restore failed")
    eligible = [entry for entry in grid if entry["eligible"]]
    if not eligible:
        (RUN_DIR / "verdict.json").write_text(json.dumps({"aborted": "no setting within limits", "grid": grid}, indent=2) + "\n")
        raise SystemExit("aborted: no setting within limits")
    chosen = max(eligible, key=lambda entry: (entry["fit_label_accuracy"], -entry["relative_frobenius_vs_tensor"]))
    delta = solve(chosen["gain"], chosen["mu"])
    print("chosen", json.dumps(chosen), flush=True)

    decision_records = [decision_record(case, arm, split_name)
                        for split_name, group in (("fit", fit), ("test", test))
                        for arm in ("original", "reversed_mapping") for case in group]
    original = evaluate_all(model, tokenizer, "original", RUN_DIR, dev, fit, test, decision_records, first_token)
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
        "relative_frobenius_vs_tensor": float(applied.norm() / tensor_norm),
        "relative_frobenius_vs_columns": float(applied.norm() / columns.norm()),
        "max_abs_change": float(applied.abs().max()), "parameter_sha256_edited": edited_hash,
    }
    torch.save({"tensor": edit_info["tensor"], "columns": units.cpu(), "edited_columns": w2.weight.data[:, units].cpu().clone()},
               RUN_DIR / "edit.pt")
    print("edit", json.dumps(edit_info), flush=True)
    edited = evaluate_all(model, tokenizer, "edited", RUN_DIR, dev, fit, test, decision_records, first_token)
    w2.weight.data.copy_(original_w2)
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

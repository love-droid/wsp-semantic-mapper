"""Frozen head-level diagnostic of the form_type first-label copy. Never edits weights.

form-type-decision-v1 found the model copies the first canonical-label mapping key and that
layer14.attention is the largest mid-stack push from the gold label toward it. Here each
attention head's direct share of that push is measured, each head is zeroed at every position
(output identical to zeroing its 64 out_proj input columns), and the source tokens the top
heads read are recorded. Development claim_form cases only.
"""
import json
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_form_type import LABELS, MAPPING_MARK, PREFIX, reversed_case
from run_fieldbench import DATA_DIR, MODEL_REVISION, ROOT, build_prompt


BATCH = 8
GREEDY_MAX_NEW = 16
HEAD_DIM = 64
ATTENTION_LAYERS = (10, 12, 14)
ABLATION_LAYER = 14
TOP_HEADS = 3
PATTERN_CASES = 24
RUN_DIR = ROOT / "experiments/form_type/heads-v1"
RULES = {
    "question": "Which attention heads copy the first mapping key at the form_type label token, and does zeroing them let the gold label win?",
    "cases": "122 development claim_form.yaml cases, original and reversed_mapping arms. Held-out not loaded.",
    "attribution": "Direct share of each head in layers 10, 12, 14 on (gold first token minus first-listed-key first token) at the label decision, scaled by the case's final RMS. Cases whose gold is the first-listed key are excluded.",
    "ablation": "Zero one layer-14 head's 64-dim output slice at every position, equal to zeroing out_proj columns [64h, 64h+64). All 16 heads singly, then the top-2 and top-3 by attribution, then the whole layer.",
    "metrics": "First-token label accuracy among the five labels, first-listed-key share, and greedy exact label after the quote for top-head ablations.",
    "patterns": "Eager attention on the first 24 original-arm cases. Top source tokens for the top heads and the share of attention on the mapping-dict span versus the document text.",
    "weights": "Frozen.",
}


def main():
    cases = [json.loads(line) for line in (DATA_DIR / "development.jsonl").open()]
    cases = [case for case in cases if case["schema_name"] == "claim_form.yaml"]
    if len(cases) != 122:
        raise AssertionError(len(cases))
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")

    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    print("parameter sha256", before, flush=True)

    quote_ids = tokenizer.encode(PREFIX + ' "', add_special_tokens=False)
    first_token = {label: tokenizer.encode(PREFIX + ' "' + label + '"', add_special_tokens=False)[len(quote_ids)]
                   for label in LABELS}
    records = []
    for arm in ("original", "reversed_mapping"):
        for case in cases:
            built = case if arm == "original" else reversed_case(case)
            mapping = json.loads(built["json_schema"]["properties"]["form_type"]["description"].split(MAPPING_MARK)[1])
            messages, _ = build_prompt(built)
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids.tolist()
            records.append({"id": case["id"], "arm": arm, "gold": case["expected"]["form_type"],
                            "first_key": next(iter(mapping)), "ids": prompt_ids + quote_ids,
                            "input": case["input"]})

    def pad(id_lists):
        encoded = tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists],
            padding=True, return_tensors="pt",
        ).to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return encoded, positions

    layers = model.model.layers
    for index in ATTENTION_LAYERS + (ABLATION_LAYER,):
        if not layers[index].is_attention_layer:
            raise AssertionError(f"layer {index} is not attention")
    started = time.perf_counter()

    head_inputs = defaultdict(list)
    final = []
    hooks = []
    for index in ATTENTION_LAYERS:
        def grab(module, args, index=index):
            head_inputs[index].append(args[0][:, -1].detach().float().clone())
        hooks.append(layers[index].self_attn.out_proj.register_forward_pre_hook(grab))
    hooks.append(layers[15].register_forward_hook(
        lambda module, args, output: final.append((output[0] if isinstance(output, tuple) else output)[:, -1].detach().float().clone())))
    base_logits = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH):
            encoded, positions = pad([record["ids"] for record in records[start:start + BATCH]])
            base_logits.append(model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float())
    for hook in hooks:
        hook.remove()
    base_logits = torch.cat(base_logits)
    final = torch.cat(final)
    head_inputs = {index: torch.cat(chunks) for index, chunks in head_inputs.items()}

    norm = model.model.embedding_norm
    gain = norm.weight.float()
    head = model.lm_head.weight.float()
    rms = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    attribution = {}
    for arm in ("original", "reversed_mapping"):
        chosen = [index for index, record in enumerate(records) if record["arm"] == arm and record["gold"] != record["first_key"]]
        for layer_index in ATTENTION_LAYERS:
            weight = layers[layer_index].self_attn.out_proj.weight.float()
            for h in range(16):
                columns = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                values = []
                for index in chosen:
                    record = records[index]
                    direction = (head[first_token[record["gold"]]] - head[first_token[record["first_key"]]]) * gain * rms[index]
                    contribution = weight[:, columns] @ head_inputs[layer_index][index, columns]
                    values.append(float(contribution @ direction))
                attribution[f"{arm}:layer{layer_index}.head{h}"] = sum(values) / len(values)
    ranked_original = sorted(
        (key for key in attribution if key.startswith(f"original:layer{ABLATION_LAYER}.")), key=lambda key: attribution[key]
    )
    top_heads = [int(key.rsplit("head", 1)[1]) for key in ranked_original[:TOP_HEADS]]
    print("top layer-14 heads pushing to first key", top_heads,
          [round(attribution[f"original:layer{ABLATION_LAYER}.head{h}"], 3) for h in top_heads], flush=True)

    def label_metrics(logits):
        result = defaultdict(lambda: {"cases": 0, "label_accuracy": 0, "first_key_choice": 0, "argmax_is_label_token": 0})
        for index, record in enumerate(records):
            scores = {label: float(logits[index, first_token[label]]) for label in LABELS}
            choice = max(LABELS, key=scores.get)
            bucket = result[record["arm"]]
            bucket["cases"] += 1
            bucket["label_accuracy"] += choice == record["gold"]
            bucket["first_key_choice"] += choice == record["first_key"]
            bucket["argmax_is_label_token"] += int(logits[index].argmax()) == first_token[record["gold"]]
        return {arm: {**values, "label_accuracy": values["label_accuracy"] / values["cases"],
                      "first_key_choice": values["first_key_choice"] / values["cases"],
                      "argmax_is_label_token": values["argmax_is_label_token"] / values["cases"]}
                for arm, values in result.items()}

    def zero_hook(heads):
        def hook(module, args):
            value = args[0].clone()
            for h in heads:
                value[..., h * HEAD_DIM:(h + 1) * HEAD_DIM] = 0
            return (value,)
        return hook

    def ablated_logits(heads):
        handle = layers[ABLATION_LAYER].self_attn.out_proj.register_forward_pre_hook(zero_hook(heads))
        try:
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(records), BATCH):
                    encoded, positions = pad([record["ids"] for record in records[start:start + BATCH]])
                    chunks.append(model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float())
            return torch.cat(chunks)
        finally:
            handle.remove()

    def greedy(heads):
        handle = layers[ABLATION_LAYER].self_attn.out_proj.register_forward_pre_hook(zero_hook(heads)) if heads else None
        texts = []
        try:
            with torch.inference_mode():
                for start in range(0, len(records), BATCH):
                    encoded, _ = pad([record["ids"] for record in records[start:start + BATCH]])
                    width = encoded["input_ids"].shape[1]
                    out = model.generate(**encoded, do_sample=False, max_new_tokens=GREEDY_MAX_NEW,
                                         repetition_penalty=1.0, use_cache=True)
                    texts.extend(tokenizer.decode(seq[width:], skip_special_tokens=True) for seq in out)
        finally:
            if handle:
                handle.remove()
        return texts

    def greedy_summary(texts):
        result = {}
        for arm in ("original", "reversed_mapping"):
            pairs = [(record, text) for record, text in zip(records, texts) if record["arm"] == arm]
            result[arm] = {
                "exact_label_accuracy": sum(text.split('"')[0] == record["gold"] for record, text in pairs) / len(pairs),
                "first_key_rate": sum(text.split('"')[0] == record["first_key"] for record, text in pairs) / len(pairs),
                "labels": Counter(text.split('"')[0] for _, text in pairs).most_common(8),
            }
        return result

    attention_index = sum(layer.is_attention_layer for layer in layers[:ABLATION_LAYER])
    zero_check = float((ablated_logits([]) - base_logits).abs().max())
    ablations = {"none": label_metrics(base_logits)}
    for h in range(16):
        ablations[f"head{h}"] = label_metrics(ablated_logits([h]))
    for count in (2, 3):
        ablations["top" + str(count)] = label_metrics(ablated_logits(top_heads[:count]))
    ablations["all_heads"] = label_metrics(ablated_logits(list(range(16))))
    print(f"ablations done {time.perf_counter() - started:.1f}s", flush=True)

    greedy_runs = {"none": greedy_summary(greedy([]))}
    for name, heads in (("top1", top_heads[:1]), ("top2", top_heads[:2]), ("top3", top_heads[:3])):
        greedy_runs[name] = greedy_summary(greedy(heads))
    print(f"greedy done {time.perf_counter() - started:.1f}s", flush=True)
    (RUN_DIR / "ablations.json").write_text(json.dumps({
        "head_attribution_gold_minus_first_key": dict(sorted(attribution.items(), key=lambda item: item[1])),
        "top_heads_layer14": top_heads, "empty_ablation_max_logit_difference": zero_check,
        "ablations_first_token": ablations, "greedy": greedy_runs,
    }, indent=2) + "\n")

    del model
    torch.cuda.empty_cache()
    eager = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    patterns = []
    originals = [record for record in records if record["arm"] == "original"][:PATTERN_CASES]
    with torch.inference_mode():
        for record in originals:
            ids = torch.tensor([record["ids"]], device="cuda")
            attention = eager(input_ids=ids, output_attentions=True, use_cache=False, logits_to_keep=1).attentions[attention_index][0, :, -1].float()
            tokens = record["ids"]
            pieces = [tokenizer.decode([token]) for token in tokens]
            starts = []
            cursor = 0
            for piece in pieces:
                starts.append(cursor)
                cursor += len(piece)
            joined = "".join(pieces)

            def find(marker):
                at = joined.rfind(marker)
                if at < 0:
                    return None
                return max(position for position, start in enumerate(starts) if start <= at)

            mapping_start = find(MAPPING_MARK.strip())
            source_start = find("SOURCE DOCUMENT TEXT")
            if mapping_start is None or source_start is None or mapping_start >= source_start:
                raise AssertionError(f"span markers not found for {record['id']}")
            for h in top_heads:
                row = attention[h]
                top = torch.topk(row, 8)
                patterns.append({
                    "id": record["id"], "head": h, "gold": record["gold"],
                    "mapping_span_share": float(row[mapping_start:source_start].sum()) if mapping_start and source_start else None,
                    "document_span_share": float(row[source_start:].sum()) if source_start else None,
                    "top_sources": [
                        {"position": int(position), "weight": float(weight),
                         "token": tokenizer.decode([tokens[int(position)]]),
                         "context": tokenizer.decode(tokens[max(0, int(position) - 4):int(position) + 3]),
                         "region": ("mapping" if mapping_start and source_start and mapping_start <= int(position) < source_start
                                    else "document" if source_start and int(position) >= source_start else "other")}
                        for weight, position in zip(top.values.tolist(), top.indices.tolist())
                    ],
                })
    after = digest_parameters(eager)
    with (RUN_DIR / "patterns.jsonl").open("w") as stream:
        for row in patterns:
            stream.write(json.dumps(row) + "\n")

    pattern_summary = {}
    for h in top_heads:
        rows = [row for row in patterns if row["head"] == h]
        pattern_summary[f"head{h}"] = {
            "mean_mapping_span_share": sum(row["mapping_span_share"] or 0 for row in rows) / len(rows),
            "mean_document_span_share": sum(row["document_span_share"] or 0 for row in rows) / len(rows),
            "top1_regions": Counter(row["top_sources"][0]["region"] for row in rows),
            "top1_tokens": Counter(row["top_sources"][0]["token"] for row in rows).most_common(8),
        }
    summary = {
        "model_revision": MODEL_REVISION,
        "run": "form-type-heads-v1",
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "modified_parameters": 0,
        "first_label_tokens": first_token,
        "head_attribution_gold_minus_first_key": dict(sorted(attribution.items(), key=lambda item: item[1])),
        "top_heads_layer14": top_heads,
        "empty_ablation_max_logit_difference": zero_check,
        "ablations_first_token": ablations,
        "greedy": greedy_runs,
        "patterns": pattern_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"ablations": {name: {arm: round(values["label_accuracy"], 3) for arm, values in result.items()}
                                    for name, result in ablations.items()},
                      "greedy": {name: {arm: values["exact_label_accuracy"] for arm, values in result.items()}
                                 for name, result in greedy_runs.items()},
                      "patterns": pattern_summary, "zero_check": zero_check, "hash_same": before == after}, indent=1), flush=True)
    if before != after or zero_check != 0:
        raise AssertionError("control failed")


if __name__ == "__main__":
    main()

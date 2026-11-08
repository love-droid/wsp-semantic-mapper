"""Frozen patch scored on the quote-versus-null token race. Never edits weights.

donor-transfer-v1 required a finished JSON null and found none. Here every absence
document is a donor. At the prefix {"<field>": the race is token 997 (' "') versus
token 3996 (' null'). A repair is 3996 winning on the absent case and 997 still
winning on the present case. literal_null stays a copying control.
"""
import json
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_null_donors import FIELD_NEXT, PRIMARY_SITE, RESIDUAL_SITE, SWEEP_SITES, build_records, score_raw
from run_fieldbench import MODEL_REVISION, ROOT, build_prompt


BATCH = 16
CONTINUATION_MAX_NEW = 96
QUOTE_ID = 997
NULL_ID = 3996
RUN_DIR = ROOT / "experiments/null_pairs/token-race-v1"
RULES = {
    "question": "Does an absence-document activation make token 3996 (' null') beat token 997 (' \"') on the paired absent case without doing so on the paired present case?",
    "donors": "All five predeclared variants from donor-transfer-v1, including documents that never emitted JSON null.",
    "not_a_successful_null_copy": "This is not evidence that a successful null representation was transferred. donor-transfer-v1 found no such representation.",
    "repair": "Absent destination's greedy token becomes 3996, and the present destination's greedy token stays outside the null family.",
    "continuation": "Generated only where the patched greedy token differs from the unpatched one. Later tokens are unpatched.",
    "sites": "layer11.ffn primary. layer13.ffn and layer10.attention are the next v2 component sites. Residual swap is an endpoint check.",
    "weights": "Frozen.",
    "held_out": "Not loaded.",
}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    records = build_records()
    (RUN_DIR / "cases.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    by_id = {record["id"]: index for index, record in enumerate(records)}
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model.requires_grad_(False)
    torch.set_num_threads(8)
    print("model loaded; hashing parameters", flush=True)
    before = digest_parameters(model)
    print("parameter sha256", before, flush=True)
    if tokenizer.encode("null", add_special_tokens=False) != [18298]:
        raise AssertionError("unexpected null token")
    if tokenizer.decode([QUOTE_ID]) != ' "' or tokenizer.decode([NULL_ID]) != " null":
        raise AssertionError((tokenizer.decode([QUOTE_ID]), tokenizer.decode([NULL_ID])))

    for record in records:
        messages, _ = build_prompt(record)
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids.tolist()
        record["decision_ids"] = prompt_ids + tokenizer.encode('{"' + record["field"] + '":', add_special_tokens=False)

    def forward(batch):
        encoded = tokenizer.pad(
            [{"input_ids": record["decision_ids"], "attention_mask": [1] * len(record["decision_ids"])} for record in batch],
            padding=True, return_tensors="pt",
        ).to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def generate_tokens(id_lists):
        outputs = []
        for start in range(0, len(id_lists), BATCH):
            chunk = id_lists[start:start + BATCH]
            encoded = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in chunk],
                padding=True, return_tensors="pt",
            ).to("cuda")
            width = encoded["input_ids"].shape[1]
            sequences = model.generate(
                **encoded, do_sample=False, max_new_tokens=CONTINUATION_MAX_NEW,
                repetition_penalty=1.0, use_cache=True,
            )
            for sequence in sequences:
                tokens = sequence[width:].tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                outputs.append(tokens)
        return outputs

    sites = {}
    for index, layer in enumerate(model.model.layers):
        sites[f"layer{index}.ffn"] = layer.feed_forward
        mixer = "attention" if layer.is_attention_layer else "conv"
        sites[f"layer{index}.{mixer}"] = layer.self_attn if layer.is_attention_layer else layer.conv
    sites[RESIDUAL_SITE] = model.model.layers[15]
    captured = {name: sites[name] for name in (*SWEEP_SITES, RESIDUAL_SITE)}
    activations = defaultdict(list)
    hooks = []
    for name, module in captured.items():
        def capture(module, args, output, name=name):
            value = output[0] if isinstance(output, tuple) else output
            activations[name].append(value[:, -1].detach().clone())
        hooks.append(module.register_forward_hook(capture))
    started = time.perf_counter()
    with torch.inference_mode():
        base_logits = torch.cat([forward(records[start:start + BATCH]) for start in range(0, len(records), BATCH)])
    for hook in hooks:
        hook.remove()
    activations = {name: torch.cat(chunks) for name, chunks in activations.items()}

    def race_gap(logits):
        return logits[:, NULL_ID] - logits[:, QUOTE_ID]

    def patched_logits(destinations, sources, site):
        chunks = []
        for start in range(0, len(destinations), BATCH):
            batch = [records[index] for index in destinations[start:start + BATCH]]
            replacement = activations[site].index_select(
                0, torch.tensor(sources[start:start + BATCH], device="cuda")
            )
            def patch(module, args, output, replacement=replacement):
                value = output[0] if isinstance(output, tuple) else output
                if value.shape[1] == 1:
                    return output
                value = value.clone()
                value[:, -1] = replacement
                return (value, *output[1:]) if isinstance(output, tuple) else value
            hook = sites[site].register_forward_hook(patch)
            try:
                chunks.append(forward(batch))
            finally:
                hook.remove()
        return torch.cat(chunks)

    donors = [record for record in records if record["state"] == "donor"]
    transfers = []
    with torch.inference_mode(), (RUN_DIR / "transfers.jsonl").open("w") as stream:
        for site in SWEEP_SITES:
            jobs = []
            for donor in donors:
                absent_id = f"{donor['pair_id']}-absent"
                present_id = f"{donor['pair_id']}-present"
                field, pair_index = donor["pair_id"].rsplit("-", 1)
                mismatch_id = f"{FIELD_NEXT[field]}-{pair_index}-donor-{donor['variant']}"
                jobs.append(("matched_absent", by_id[donor["id"]], by_id[absent_id], donor["id"]))
                jobs.append(("matched_present", by_id[donor["id"]], by_id[present_id], donor["id"]))
                jobs.append(("mismatched_absent", by_id[mismatch_id], by_id[absent_id], mismatch_id))
                jobs.append(("mismatched_present", by_id[mismatch_id], by_id[present_id], mismatch_id))
                jobs.append(("reverse_absent_into_donor", by_id[absent_id], by_id[donor["id"]], absent_id))
            destinations = [job[2] for job in jobs]
            sources = [job[1] for job in jobs]
            logits = patched_logits(destinations, sources, site)
            gaps = race_gap(logits)
            changed = [offset for offset, destination in enumerate(destinations)
                       if int(logits[offset].argmax()) != int(base_logits[destination].argmax())]
            continuations = {}
            if changed and site == PRIMARY_SITE:
                first = [int(logits[offset].argmax()) for offset in changed]
                rests = generate_tokens(
                    [records[destinations[offset]]["decision_ids"] + [token] for offset, token in zip(changed, first)]
                )
                for offset, token, rest in zip(changed, first, rests):
                    text = tokenizer.decode([token] + rest, skip_special_tokens=True)
                    full = '{"' + records[destinations[offset]]["field"] + '":' + text
                    continuations[offset] = score_raw(records[destinations[offset]], full)
                    continuations[offset]["continuation"] = text
            for offset, job in enumerate(jobs):
                kind, source, destination, source_name = job
                top = int(logits[offset].argmax())
                base_top = int(base_logits[destination].argmax())
                record = {
                    "site": site, "kind": kind,
                    "source_id": source_name, "destination_id": records[destination]["id"],
                    "destination_state": records[destination]["state"],
                    "variant": records[source]["variant"], "donor_class": records[source]["donor_class"],
                    "field": records[destination]["field"],
                    "unpatched_gap": float(race_gap(base_logits[destination:destination + 1])[0]),
                    "patched_gap": float(gaps[offset]),
                    "gap_delta": float(gaps[offset] - race_gap(base_logits[destination:destination + 1])[0]),
                    "unpatched_top": base_top, "patched_top": top,
                    "unpatched_top_text": tokenizer.decode([base_top]),
                    "patched_top_text": tokenizer.decode([top]),
                    "flipped_to_null_token": base_top != NULL_ID and top == NULL_ID,
                    "continuation": continuations.get(offset),
                }
                transfers.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            flips = sum(row["flipped_to_null_token"] for row in transfers if row["site"] == site)
            print(f"{site} jobs {len(jobs)} flipped_to_null {flips}", flush=True)

        own = patched_logits(list(range(len(records))), list(range(len(records))), PRIMARY_SITE)
        self_error = float((own - base_logits).abs().max())
        rolled = [(index + 1) % len(records) for index in range(len(records))]
        residual = patched_logits(list(range(len(records))), rolled, RESIDUAL_SITE)
        residual_error = float((residual - base_logits[rolled]).abs().max())
        print(f"self-patch {self_error} residual {residual_error}", flush=True)

    def pack(rows):
        absent = [row for row in rows if row["kind"].endswith("absent") or row["destination_state"] == "absent"]
        present = [row for row in rows if row["destination_state"] == "present"]
        return {
            "cases": len(rows),
            "mean_gap_delta": sum(row["gap_delta"] for row in rows) / len(rows),
            "mean_patched_gap": sum(row["patched_gap"] for row in rows) / len(rows),
            "flipped_to_null_token": sum(row["flipped_to_null_token"] for row in rows),
            "absent_cases": len(absent),
            "absent_flipped_to_null_token": sum(row["flipped_to_null_token"] for row in absent),
            "absent_mean_gap_delta": (sum(row["gap_delta"] for row in absent) / len(absent)) if absent else None,
            "present_cases": len(present),
            "present_flipped_to_null_token": sum(row["flipped_to_null_token"] for row in present),
            "present_mean_gap_delta": (sum(row["gap_delta"] for row in present) / len(present)) if present else None,
            "top_token_changed": sum(row["patched_top"] != row["unpatched_top"] for row in rows),
        }

    grouped = defaultdict(list)
    for row in transfers:
        grouped[(row["donor_class"], row["variant"], row["kind"], row["site"])].append(row)
    base_gap = race_gap(base_logits)
    baseline = defaultdict(list)
    for index, record in enumerate(records):
        baseline[f"{record['donor_class']}:{record['variant']}:{record['state']}"].append(float(base_gap[index]))
    after = digest_parameters(model)
    summary = {
        "model_revision": MODEL_REVISION,
        "run": "token-race-v1",
        "quote_token_id": QUOTE_ID,
        "null_race_token_id": NULL_ID,
        "quote_text": tokenizer.decode([QUOTE_ID]),
        "null_race_text": tokenizer.decode([NULL_ID]),
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "modified_parameters": 0,
        "self_patch_max_logit_difference": self_error,
        "residual_swap_max_logit_difference": residual_error,
        "unpatched_gap_null_minus_quote": {
            key: {"cases": len(values), "min": min(values), "median": sorted(values)[len(values) // 2], "max": max(values)}
            for key, values in sorted(baseline.items())
        },
        "transfers": {
            f"{donor_class}:{variant}:{kind}:{site}": pack(rows)
            for (donor_class, variant, kind, site), rows in sorted(grouped.items())
        },
        "elapsed_seconds": time.perf_counter() - started,
        "caveat": "Pairwise token gap, not the v2 token-family margin. Patching a document that itself emits a quote cannot demonstrate recovery from a successful null. No weight patch is selected.",
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "flipped_to_null_token": sum(row["flipped_to_null_token"] for row in transfers),
        "top_token_changed": sum(row["patched_top"] != row["unpatched_top"] for row in transfers),
        "elapsed_seconds": summary["elapsed_seconds"],
        "self_patch": self_error,
        "residual": residual_error,
        "parameter_sha256": after,
    }, indent=2), flush=True)
    if before != after or self_error != 0 or residual_error != 0:
        raise AssertionError(f"control failed hash_same={before == after} self={self_error} residual={residual_error}")


if __name__ == "__main__":
    main()

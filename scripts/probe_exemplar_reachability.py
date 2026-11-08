"""Frozen reachability battery. Never edits weights.

The 24 synthetic pairs never emit JSON null under the original prompt, and absence
activations push present values toward null. This run asks whether a correct null
is reachable at all, using three prompt changes fixed before the run:

- schema_says_json_null: append one sentence to the existing field description
- null_exemplar_only: one fixed example whose target is JSON null
- two_exemplars: that null example plus one fixed example whose target is a real string

The original prompt is regenerated in the same run. Examples use none of the test
values. A pair counts only when the absent target is JSON null, the present target
stays exact, and record_id and insurer stay exact on both. If any changed-prompt
pair succeeds, its absent-decision activation is patched into the original-prompt
pair at layer 11 feed-forward and layer 10 attention. Held-out data is not loaded.
"""
import copy
import json
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_null_donors import FIELD_NEXT, score_raw
from probe_null_pairs import make_cases
from run_fieldbench import MODEL_REVISION, ROOT, build_prompt


BATCH = 16
FREE_MAX_NEW = 256
CONTINUATION_MAX_NEW = 96
QUOTE_ID = 997
NULL_ID = 3996
RUN_DIR = ROOT / "experiments/null_pairs/exemplar-reachability-v1"
SITES = ("layer11.ffn", "layer10.attention")
RESIDUAL_SITE = "layer15.residual_control"
SCHEMA_NOTE = (
    " If the source does not state this value, output a JSON null, not a quoted string and not an empty string."
)
ARMS = ("original", "schema_says_json_null", "null_exemplar_only", "two_exemplars")
EXEMPLAR_FIELDS = {
    "insured_name": ("name", "Named insured", "Harbor Glue Co", "Broker: Example Brokerage"),
    "policy_number": ("pol", "Policy number", "EX-1001", "Quote reference: EX-QUOTE"),
    "loss_date": ("date", "Date of loss", "2019-01-02", "Report prepared: 2019-02-02"),
}
RULES = {
    "arms": "original, schema_says_json_null, null_exemplar_only, two_exemplars. Fixed before scoring.",
    "examples": "Northwind Assurance, Harbor Glue Co, EX-1001, 2019-01-02, REC-EX-*. Same examples for present and absent test cases. Present example is shown before the absent example.",
    "pair_success": "Absent target is JSON null, present target is the exact source string, both keep record_id and insurer, both schema-valid. Empty string and quoted null do not count.",
    "transfer_gate": "Prefer free-generation pair successes. If none, use decision-prefix continuation pair successes. Otherwise do not transfer.",
    "transfer": "Patch the successful absent prompt's decision activation into the original-prompt absent case and the original-prompt present case.",
    "repair": "Original absent greedy token becomes 3996 and original present greedy token does not.",
    "weights": "Frozen. A positive transfer is not permission to edit weights.",
    "held_out": "Not loaded.",
    "stop": "No further prompt variants are added after seeing these results.",
}


def exemplar_docs(field):
    short, label, value, distractor = EXEMPLAR_FIELDS[field]
    insurer = "Northwind Assurance"
    present_doc = "\n".join([
        "Insurance record", f"Record ID: REC-EX-{short}-A", f"Insurer: {insurer}", f"{label}: {value}",
    ])
    absent_doc = "\n".join([
        "Insurance record", f"Record ID: REC-EX-{short}-B", f"Insurer: {insurer}", distractor,
    ])
    present_json = {field: value, "record_id": f"REC-EX-{short}-A", "insurer": insurer}
    absent_json = {field: None, "record_id": f"REC-EX-{short}-B", "insurer": insurer}
    if value in absent_doc:
        raise AssertionError(field)
    return present_doc, present_json, absent_doc, absent_json


def example_text(doc, payload):
    return "Document:\n" + doc + "\nJSON:\n" + json.dumps(payload, indent=2)


def make_records():
    records = []
    for case in make_cases():
        present_doc, present_json, absent_doc, absent_json = exemplar_docs(case["field"])
        blocks = {
            "null_exemplar_only": example_text(absent_doc, absent_json),
            "two_exemplars": example_text(present_doc, present_json) + "\n\n" + example_text(absent_doc, absent_json),
        }
        for arm in ARMS:
            material = case if arm != "schema_says_json_null" else copy.deepcopy(case)
            if arm == "schema_says_json_null":
                prop = material["json_schema"]["properties"][material["field"]]
                if SCHEMA_NOTE.strip() in prop["description"]:
                    raise AssertionError("schema note already present")
                prop["description"] = prop["description"] + SCHEMA_NOTE
            messages, _ = build_prompt(material)
            if arm in blocks:
                needle = "Extract this document into the target schema.\n\nTARGET JSON SCHEMA:\n"
                insert = "Extract this document into the target schema.\n\nEXAMPLES:\n" + blocks[arm] + "\n\nTARGET JSON SCHEMA:\n"
                messages[1]["content"] = messages[1]["content"].replace(needle, insert, 1)
                if messages[1]["content"].count("EXAMPLES:") != 1:
                    raise AssertionError(arm)
            if any(token in messages[1]["content"] and arm == "original" for token in ("Harbor Glue Co", "EX-1001", "Northwind")):
                raise AssertionError("example leaked into original")
            records.append({
                "id": f"{case['id']}__{arm}",
                "base_id": case["id"],
                "pair_id": case["pair_id"],
                "state": case["state"],
                "field": case["field"],
                "index": int(case["pair_id"].rsplit("-", 1)[1]),
                "arm": arm,
                "input": case["input"],
                "json_schema": material["json_schema"],
                "expected": case["expected"],
                "messages": messages,
            })
    if len(records) != 192:
        raise AssertionError(len(records))
    return records


def pair_success(absent, present):
    absent_ok = absent["target_is_null"] and absent["retained_fields_correct"] and absent["schema_valid"]
    present_ok = present["target_correct"] and present["retained_fields_correct"] and present["schema_valid"]
    return absent_ok and present_ok


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(RULES, indent=2) + "\n")
    records = make_records()
    stored = [{key: value for key, value in record.items() if key != "messages"} for record in records]
    (RUN_DIR / "cases.jsonl").write_text("".join(json.dumps(record) + "\n" for record in stored))
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
    if tokenizer.decode([QUOTE_ID]) != ' "' or tokenizer.decode([NULL_ID]) != " null":
        raise AssertionError("unexpected race tokens")

    for record in records:
        prompt_ids = tokenizer.apply_chat_template(
            record["messages"], tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids.tolist()
        record["prompt_ids"] = prompt_ids
        record["decision_ids"] = prompt_ids + tokenizer.encode('{"' + record["field"] + '":', add_special_tokens=False)

    def generate_tokens(id_lists, max_new):
        outputs = []
        for start in range(0, len(id_lists), BATCH):
            chunk = id_lists[start:start + BATCH]
            encoded = tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in chunk],
                padding=True, return_tensors="pt",
            ).to("cuda")
            width = encoded["input_ids"].shape[1]
            sequences = model.generate(
                **encoded, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, use_cache=True,
            )
            for sequence in sequences:
                tokens = sequence[width:].tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                outputs.append(tokens)
        return outputs

    def forward(batch):
        encoded = tokenizer.pad(
            [{"input_ids": record["decision_ids"], "attention_mask": [1] * len(record["decision_ids"])} for record in batch],
            padding=True, return_tensors="pt",
        ).to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()

    def groups(arm):
        return [record for record in records if record["arm"] == arm]

    started = time.perf_counter()
    generations = {}
    with torch.inference_mode():
        for arm in ARMS:
            group = groups(arm)
            token_lists = generate_tokens([record["prompt_ids"] for record in group], FREE_MAX_NEW)
            for record, tokens in zip(group, token_lists):
                scored = score_raw(record, tokenizer.decode(tokens, skip_special_tokens=True))
                scored.update({"id": record["id"], "arm": arm, "state": record["state"], "base_id": record["base_id"]})
                generations[record["id"]] = scored
            absent = [generations[record["id"]] for record in group if record["state"] == "absent"]
            present = [generations[record["id"]] for record in group if record["state"] == "present"]
            print(
                f"free {arm} absent_null {sum(row['target_is_null'] for row in absent)}/{len(absent)}"
                f" present_correct {sum(row['target_correct'] for row in present)}/{len(present)}",
                flush=True,
            )
    (RUN_DIR / "generations.jsonl").write_text("".join(json.dumps(generations[record["id"]]) + "\n" for record in records))

    sites = {}
    for index, layer in enumerate(model.model.layers):
        sites[f"layer{index}.ffn"] = layer.feed_forward
        mixer = "attention" if layer.is_attention_layer else "conv"
        sites[f"layer{index}.{mixer}"] = layer.self_attn if layer.is_attention_layer else layer.conv
    sites[RESIDUAL_SITE] = model.model.layers[15]
    activations = defaultdict(list)
    hooks = []
    for name in (*SITES, RESIDUAL_SITE):
        def capture(module, args, output, name=name):
            value = output[0] if isinstance(output, tuple) else output
            activations[name].append(value[:, -1].detach().clone())
        hooks.append(sites[name].register_forward_hook(capture))
    with torch.inference_mode():
        base_logits = torch.cat([forward(records[start:start + BATCH]) for start in range(0, len(records), BATCH)])
    for hook in hooks:
        hook.remove()
    activations = {name: torch.cat(chunks) for name, chunks in activations.items()}
    gap = base_logits[:, NULL_ID] - base_logits[:, QUOTE_ID]

    decisions = {}
    with torch.inference_mode():
        for arm in ARMS:
            group = groups(arm)
            indices = [by_id[record["id"]] for record in group]
            first = [int(base_logits[index].argmax()) for index in indices]
            rests = generate_tokens(
                [record["decision_ids"] + [token] for record, token in zip(group, first)],
                CONTINUATION_MAX_NEW,
            )
            for record, token, rest in zip(group, first, rests):
                text = tokenizer.decode([token] + rest, skip_special_tokens=True)
                scored = score_raw(record, '{"' + record["field"] + '":' + text)
                index = by_id[record["id"]]
                scored.update({
                    "id": record["id"], "arm": arm, "state": record["state"], "base_id": record["base_id"],
                    "top_token": token, "top_text": tokenizer.decode([token]),
                    "gap_null_minus_quote": float(gap[index]),
                    "continuation": text,
                })
                decisions[record["id"]] = scored
            print(f"prefix {arm}", flush=True)
    (RUN_DIR / "decisions.jsonl").write_text("".join(json.dumps(decisions[record["id"]]) + "\n" for record in records))

    def successful_pairs(table):
        found = []
        for arm in ARMS:
            if arm == "original":
                continue
            for pair_id in sorted({record["pair_id"] for record in records}):
                absent = table[f"{pair_id}-absent__{arm}"]
                present = table[f"{pair_id}-present__{arm}"]
                if pair_success(absent, present):
                    found.append((arm, pair_id))
        return found

    free_pairs = successful_pairs(generations)
    prefix_pairs = successful_pairs(decisions)
    chosen = free_pairs or prefix_pairs
    chosen_source = "free_generation" if free_pairs else ("prefix_continuation" if prefix_pairs else "none")
    print(f"pair successes free {len(free_pairs)} prefix {len(prefix_pairs)} using {chosen_source}", flush=True)

    def patched_logits(destinations, sources, site):
        chunks = []
        for start in range(0, len(destinations), BATCH):
            batch = [records[index] for index in destinations[start:start + BATCH]]
            replacement = activations[site].index_select(0, torch.tensor(sources[start:start + BATCH], device="cuda"))
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
        return torch.cat(chunks) if chunks else base_logits[:0]

    transfers = []
    with torch.inference_mode(), (RUN_DIR / "transfers.jsonl").open("w") as stream:
        jobs = []
        for arm, pair_id in chosen:
            field, pair_index = pair_id.rsplit("-", 1)
            source_id = f"{pair_id}-absent__{arm}"
            mismatch_id = f"{FIELD_NEXT[field]}-{pair_index}-absent__{arm}"
            mismatch_pair = f"{FIELD_NEXT[field]}-{pair_index}"
            for state in ("absent", "present"):
                destination_id = f"{pair_id}-{state}__original"
                jobs.append(("matched", source_id, destination_id))
                if (arm, mismatch_pair) in chosen:
                    jobs.append(("mismatched", mismatch_id, destination_id))
        for site in SITES:
            if not jobs:
                print(f"{site} no donors", flush=True)
                continue
            destinations = [by_id[job[2]] for job in jobs]
            sources = [by_id[job[1]] for job in jobs]
            logits = patched_logits(destinations, sources, site)
            for offset, job in enumerate(jobs):
                kind, source_id, destination_id = job
                destination = by_id[destination_id]
                top = int(logits[offset].argmax())
                base_top = int(base_logits[destination].argmax())
                patched_gap = float(logits[offset, NULL_ID] - logits[offset, QUOTE_ID])
                record = {
                    "site": site, "kind": kind, "source_id": source_id, "destination_id": destination_id,
                    "arm": source_id.rsplit("__", 1)[1], "state": records[destination]["state"],
                    "unpatched_gap": float(gap[destination]), "patched_gap": patched_gap,
                    "gap_delta": patched_gap - float(gap[destination]),
                    "unpatched_top": base_top, "patched_top": top,
                    "unpatched_top_text": tokenizer.decode([base_top]),
                    "patched_top_text": tokenizer.decode([top]),
                    "flipped_to_null_token": base_top != NULL_ID and top == NULL_ID,
                }
                transfers.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{site} transfers {len(jobs)} flips {sum(row['flipped_to_null_token'] for row in transfers if row['site'] == site)}", flush=True)

        own = patched_logits(list(range(len(records))), list(range(len(records))), SITES[0])
        self_error = float((own - base_logits).abs().max())
        rolled = [(index + 1) % len(records) for index in range(len(records))]
        residual = patched_logits(list(range(len(records))), rolled, RESIDUAL_SITE)
        residual_index = torch.tensor(rolled, device=base_logits.device)
        residual_error = float((residual - base_logits.index_select(0, residual_index)).abs().max())
        print(f"self-patch {self_error} residual {residual_error}", flush=True)

    def arm_counts(table, state, flag):
        return {
            arm: sum(table[record["id"]][flag] for record in groups(arm) if record["state"] == state)
            for arm in ARMS
        }

    summary = {
        "model_revision": MODEL_REVISION,
        "run": "exemplar-reachability-v1",
        "parameter_sha256_before": before,
        "parameter_sha256_after": digest_parameters(model),
        "modified_parameters": 0,
        "self_patch_max_logit_difference": self_error,
        "residual_swap_max_logit_difference": residual_error,
        "free_absent_json_null": arm_counts(generations, "absent", "target_is_null"),
        "free_absent_schema_and_retained_null": {
            arm: sum(
                generations[record["id"]]["target_is_null"] and generations[record["id"]]["retained_fields_correct"]
                and generations[record["id"]]["schema_valid"]
                for record in groups(arm) if record["state"] == "absent"
            )
            for arm in ARMS
        },
        "free_present_target_correct": arm_counts(generations, "present", "target_correct"),
        "prefix_absent_top_is_null_token": {
            arm: sum(decisions[record["id"]]["top_token"] == NULL_ID for record in groups(arm) if record["state"] == "absent")
            for arm in ARMS
        },
        "prefix_present_top_is_null_token": {
            arm: sum(decisions[record["id"]]["top_token"] == NULL_ID for record in groups(arm) if record["state"] == "present")
            for arm in ARMS
        },
        "prefix_absent_continuation_null_complete": arm_counts(decisions, "absent", "target_is_null"),
        "prefix_present_continuation_correct": arm_counts(decisions, "present", "target_correct"),
        "free_generation_pair_successes": [list(item) for item in free_pairs],
        "prefix_pair_successes": [list(item) for item in prefix_pairs],
        "transfer_source": chosen_source,
        "transfer_flips_to_null_token": sum(row["flipped_to_null_token"] for row in transfers),
        "elapsed_seconds": time.perf_counter() - started,
        "caveat": "Prompt changes are reachability tests, not weight edits. Transfer into the original prompt is the weight-relevant result. No weights were modified.",
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "free_pairs": len(free_pairs),
        "prefix_pairs": len(prefix_pairs),
        "transfer_source": chosen_source,
        "transfer_flips": summary["transfer_flips_to_null_token"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "parameter_sha256": summary["parameter_sha256_after"],
    }, indent=2), flush=True)
    if summary["parameter_sha256_before"] != summary["parameter_sha256_after"] or self_error != 0 or residual_error != 0:
        raise AssertionError("control failed")


if __name__ == "__main__":
    main()

"""Find successful-null donors and test frozen activation transfer. Never edits weights.

Donors are a predeclared rewrite of the 24 absent synthetic pairs from probe_null_pairs.py.
The prompt, schema, and retained facts stay fixed. literal_null is a copying control.
Held-out FieldBench cases are not read. No gradients and no parameter updates.
"""
import json
import re
import time
from collections import defaultdict

import torch
from jsonschema import Draft202012Validator, FormatChecker
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from probe_null_pairs import make_cases
from rescore_baseline import parse_fenced_json
from run_fieldbench import MODEL_REVISION, ROOT, build_prompt


BATCH = 16
FREE_MAX_NEW = 256
CONTINUATION_MAX_NEW = 96
RUN_DIR = ROOT / "experiments/null_pairs/donor-transfer-v1"
PRIMARY_SITE = "layer11.ffn"
SWEEP_SITES = (PRIMARY_SITE, "layer13.ffn", "layer10.attention")
RESIDUAL_SITE = "layer15.residual_control"
FIELD_NEXT = {"insured_name": "policy_number", "policy_number": "loss_date", "loss_date": "insured_name"}
NATURAL = {
    "insured_name": "The named insured is not stated in this record.",
    "policy_number": "The current policy number is not stated in this record.",
    "loss_date": "The date of loss is not stated in this record.",
}
VARIANTS = (
    ("explicit_not_provided", "candidate"),
    ("explicit_unknown", "candidate"),
    ("natural_absence", "candidate"),
    ("no_distractors", "candidate"),
    ("literal_null", "copying_control"),
)
SELECTION_RULES = {
    "documents": "Same schema and extraction prompt as activation-probe-v2. Only the source text changes.",
    "explicit_not_provided": "Replace the removed fact line with '<same label>: not provided'. Other lines stay.",
    "explicit_unknown": "Replace the removed fact line with '<same label>: unknown'. Other lines stay.",
    "natural_absence": "Replace the removed fact line with one fixed sentence naming that fact as unstated.",
    "no_distractors": "Absent text with quote, previous-policy, broker, date, and contact distractors removed.",
    "literal_null": "Replace the removed fact line with '<same label>: null'. Copying control, not evidence of understanding.",
    "free_generation_success": "After at most one enclosing Markdown fence, the object has JSON null at the target key and exact record_id and insurer. Empty string is not null. A missing key is not null.",
    "strict_success": "free_generation_success and the object validates against the case schema.",
    "prefix": "Assistant prefix is {\"<field>\": for every case. Null-versus-quote log mass uses the v2 token families. Greedy top token and the unpatched continuation are saved separately from that margin.",
    "primary_donor": "strict_success whose greedy decision-prefix token is in the null family. Headline transfer rates use this set. Broader free-generation successes are still transferred and labeled.",
    "mismatch": "Same variant and pair index, on the next field in insured_name -> policy_number -> loss_date. No other substitute.",
    "sweep": "layer11.ffn is the v2 primary component site. layer13.ffn and layer10.attention are the next two component sites by absolute bidirectional margin effect. Continuations are run for matched and mismatched transfers at these three sites.",
    "patch": "Only the final-position activation at the decision prefix is replaced, then the first new token is the greedy argmax of those logits. Later tokens are unpatched greedy generation.",
    "residual_control": "Swapping the final layer residual must reproduce the source next-token logits. This checks the hook, not a candidate edit.",
    "weights": "Frozen. Parameter SHA256 is recorded before and after.",
    "held_out": "The 49 held-out FieldBench cases are not loaded.",
}


def kept_line(line):
    return line in ("Insurance record", "Insurer: Atlas Mutual") or line.startswith("Record ID: ")


def make_donor(present, absent, variant, donor_class):
    field = present["field"]
    present_lines = present["input"].split("\n")
    absent_lines = absent["input"].split("\n")
    removed = [line for line in present_lines if line not in absent_lines]
    if len(removed) != 1:
        raise AssertionError(f"{present['id']} removed {removed}")
    target_line = removed[0]
    value = present["expected"][field]
    if not target_line.endswith(f": {value}"):
        raise AssertionError(target_line)
    label = target_line[: -(len(value) + 2)]
    if variant == "no_distractors":
        lines = [line for line in absent_lines if kept_line(line)]
        if len(lines) != 3:
            raise AssertionError(lines)
    else:
        replacement = {
            "explicit_not_provided": f"{label}: not provided",
            "explicit_unknown": f"{label}: unknown",
            "natural_absence": NATURAL[field],
            "literal_null": f"{label}: null",
        }[variant]
        lines = list(present_lines)
        lines[lines.index(target_line)] = replacement
        unchanged = [line for line in lines if line != replacement]
        if unchanged != [line for line in present_lines if line != target_line]:
            raise AssertionError(present["id"])
    text = "\n".join(lines)
    if value in text or text in (present["input"], absent["input"]):
        raise AssertionError(present["pair_id"] + " " + variant)
    expected = dict(absent["expected"])
    if expected[field] is not None:
        raise AssertionError(expected)
    return {
        "id": f"{present['pair_id']}-donor-{variant}",
        "pair_id": present["pair_id"],
        "state": "donor",
        "field": field,
        "index": int(present["pair_id"].rsplit("-", 1)[1]),
        "variant": variant,
        "donor_class": donor_class,
        "input": text,
        "json_schema": present["json_schema"],
        "expected": expected,
    }


def build_records():
    originals = make_cases()
    by_pair = defaultdict(dict)
    order = []
    for case in originals:
        by_pair[case["pair_id"]][case["state"]] = case
        if case["pair_id"] not in order:
            order.append(case["pair_id"])
    records = []
    for case in originals:
        records.append({
            **case,
            "variant": "original",
            "donor_class": "recipient",
            "index": int(case["pair_id"].rsplit("-", 1)[1]),
        })
    for variant, donor_class in VARIANTS:
        for pair_id in order:
            records.append(make_donor(by_pair[pair_id]["present"], by_pair[pair_id]["absent"], variant, donor_class))
    if len(records) != 168 or len({record["id"] for record in records}) != 168:
        raise AssertionError(len(records))
    return records


def score_parsed(case, parsed, parse_error):
    is_object = isinstance(parsed, dict)
    schema_errors = []
    if is_object:
        validator = Draft202012Validator(case["json_schema"], format_checker=FormatChecker())
        schema_errors = list(validator.iter_errors(parsed))
    present = is_object and case["field"] in parsed
    value = parsed[case["field"]] if present else None
    retained = is_object and all(parsed.get(key) == case["expected"][key] for key in ("record_id", "insurer"))
    return {
        "parsed": parsed,
        "parse_error": parse_error,
        "schema_valid": is_object and not schema_errors,
        "schema_error": None if not schema_errors else schema_errors[0].message,
        "target_key_present": present,
        "target_value": value,
        "target_is_null": present and value is None,
        "target_correct": present and value == case["expected"][case["field"]],
        "retained_fields_correct": retained,
        "missing_required": [key for key in case["json_schema"]["required"] if not (is_object and key in parsed)],
    }


def score_raw(case, raw):
    parsed, error, fence = parse_fenced_json(raw)
    scored = score_parsed(case, parsed, error)
    scored["fence_removed"] = fence
    return scored


def success_flags(case, scored):
    free = case["donor_class"] != "recipient" and scored["target_is_null"] and scored["retained_fields_correct"]
    return {"free_generation_success": free, "strict_success": free and scored["schema_valid"]}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "selection_rules.json").write_text(json.dumps(SELECTION_RULES, indent=2) + "\n")
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
    null_id = tokenizer.encode("null", add_special_tokens=False)
    quote_id = tokenizer.encode('"', add_special_tokens=False)
    if null_id != [18298] or quote_id != [511]:
        raise AssertionError((null_id, quote_id))
    decoded = [tokenizer.decode([token]).lstrip() for token in range(len(tokenizer))]
    null_family = [token for token, text in enumerate(decoded) if re.match(r"^null(?:$|[\s,}])", text)]
    quote_family = [token for token, text in enumerate(decoded) if text.startswith('"')]
    if set(null_family) != {3996, 18298}:
        raise AssertionError(null_family)

    def margin_of(logits):
        return torch.logsumexp(logits[:, null_family], dim=-1) - torch.logsumexp(logits[:, quote_family], dim=-1)

    for record in records:
        messages, _ = build_prompt(record)
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
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
                **encoded, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, use_cache=True
            )
            for sequence in sequences:
                tokens = sequence[width:].tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                outputs.append(tokens)
        return outputs

    def grouped(predicate):
        buckets = defaultdict(list)
        order = []
        for record in records:
            key = predicate(record)
            if key not in buckets:
                order.append(key)
            buckets[key].append(record)
        return [buckets[key] for key in order]

    started = time.perf_counter()
    generations = [None] * len(records)
    with torch.inference_mode():
        for group in grouped(lambda record: (record["donor_class"], record["variant"])):
            texts = [tokenizer.decode(tokens, skip_special_tokens=True)
                     for tokens in generate_tokens([record["prompt_ids"] for record in group], FREE_MAX_NEW)]
            for record, raw in zip(group, texts):
                scored = score_raw(record, raw)
                scored.update({"id": record["id"], "variant": record["variant"], "donor_class": record["donor_class"],
                               "state": record["state"], "raw": raw})
                scored.update(success_flags(record, scored))
                generations[by_id[record["id"]]] = scored
            print(f"free generation {group[0]['variant']} {len(group)}", flush=True)
    (RUN_DIR / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generations))

    def forward(batch):
        encoded = tokenizer.pad(
            [{"input_ids": record["decision_ids"], "attention_mask": [1] * len(record["decision_ids"])} for record in batch],
            padding=True, return_tensors="pt",
        ).to("cuda")
        positions = encoded["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(encoded["attention_mask"] == 0, 0)
        return model(**encoded, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()

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
    with torch.inference_mode():
        base_logits = torch.cat([forward(records[start:start + BATCH]) for start in range(0, len(records), BATCH)])
    for hook in hooks:
        hook.remove()
    activations = {name: torch.cat(chunks) for name, chunks in activations.items()}
    if base_logits.shape[0] != len(records) or any(value.shape[0] != len(records) for value in activations.values()):
        raise AssertionError("activation rows do not match records")
    base_margin = margin_of(base_logits)

    def top_record(logits_row):
        values, tokens = logits_row.topk(5)
        return [{"id": token, "text": tokenizer.decode([token]), "logit": value}
                for token, value in zip(tokens.tolist(), values.tolist())]

    def stitch_score(record, continuation):
        scored = score_raw(record, '{"' + record["field"] + '":' + continuation)
        scored["continuation"] = continuation
        return scored

    decision_rows = []
    with torch.inference_mode():
        for group in grouped(lambda record: (record["donor_class"], record["variant"])):
            indices = [by_id[record["id"]] for record in group]
            first = base_logits[indices].argmax(-1).tolist()
            rests = generate_tokens(
                [record["decision_ids"] + [token] for record, token in zip(group, first)],
                CONTINUATION_MAX_NEW,
            )
            for record, token, rest in zip(group, first, rests):
                continuation = tokenizer.decode([token] + rest, skip_special_tokens=True)
                row = {
                    "id": record["id"], "variant": record["variant"], "donor_class": record["donor_class"],
                    "state": record["state"], "null_minus_quote": float(base_margin[by_id[record["id"]]]),
                    "top5": top_record(base_logits[by_id[record["id"]]]),
                    "top_in_null_family": token in null_family, "top_in_quote_family": token in quote_family,
                }
                row.update(stitch_score(record, continuation))
                decision_rows.append(row)
            print(f"prefix continuation {group[0]['variant']} {len(group)}", flush=True)
    decision_by_id = {row["id"]: row for row in decision_rows}
    (RUN_DIR / "decision_baseline.json").write_text(json.dumps(decision_rows, indent=2) + "\n")

    def is_primary(record):
        generated = generations[by_id[record["id"]]]
        decided = decision_by_id[record["id"]]
        return generated["strict_success"] and decided["top_in_null_family"]

    donors = [record for record in records if generations[by_id[record["id"]]]["free_generation_success"]]
    print(f"free-generation successes {len(donors)}; primary {sum(is_primary(record) for record in donors)}", flush=True)

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
        return torch.cat(chunks) if chunks else base_logits[:0]

    def continuation_scores(destinations, logits):
        first = logits.argmax(-1).tolist()
        rests = generate_tokens(
            [records[index]["decision_ids"] + [token] for index, token in zip(destinations, first)],
            CONTINUATION_MAX_NEW,
        )
        scored = []
        for index, token, rest in zip(destinations, first, rests):
            text = tokenizer.decode([token] + rest, skip_special_tokens=True)
            row = stitch_score(records[index], text)
            row["top_token"] = token
            row["top_text"] = tokenizer.decode([token])
            row["top_in_null_family"] = token in null_family
            scored.append(row)
        return scored

    transfers = []
    with torch.inference_mode(), (RUN_DIR / "transfers.jsonl").open("w") as stream:
        for site in SWEEP_SITES:
            matched_dest, matched_source, matched_meta = [], [], []
            mismatched_dest, mismatched_source, mismatched_meta = [], [], []
            reverse_dest, reverse_source = [], []
            unavailable_mismatch = 0
            for donor in donors:
                absent_id = f"{donor['pair_id']}-absent"
                present_id = f"{donor['pair_id']}-present"
                other_field, pair_index = donor["pair_id"].rsplit("-", 1)
                mismatch_id = f"{FIELD_NEXT[other_field]}-{pair_index}-donor-{donor['variant']}"
                for destination_id, kind, dest_bucket, source_bucket, meta_bucket in (
                    (absent_id, "matched_absent", matched_dest, matched_source, matched_meta),
                    (present_id, "matched_present", matched_dest, matched_source, matched_meta),
                ):
                    dest_bucket.append(by_id[destination_id])
                    source_bucket.append(by_id[donor["id"]])
                    meta_bucket.append((kind, donor, destination_id))
                if generations[by_id[mismatch_id]]["free_generation_success"]:
                    for destination_id, kind in ((absent_id, "mismatched_absent"), (present_id, "mismatched_present")):
                        mismatched_dest.append(by_id[destination_id])
                        mismatched_source.append(by_id[mismatch_id])
                        mismatched_meta.append((kind, donor, destination_id, mismatch_id))
                else:
                    unavailable_mismatch += 1
                reverse_dest.append(by_id[donor["id"]])
                reverse_source.append(by_id[absent_id])
            plans = (
                ("matched", matched_dest, matched_source, True),
                ("mismatched", mismatched_dest, mismatched_source, True),
                ("reverse", reverse_dest, reverse_source, False),
            )
            for plan_name, destinations, sources, with_continuation in plans:
                if not destinations:
                    continue
                logits = patched_logits(destinations, sources, site)
                margins = margin_of(logits)
                if with_continuation:
                    continued = continuation_scores(destinations, logits)
                    unpatched = continuation_scores(destinations, base_logits[destinations])
                else:
                    continued = [None] * len(destinations)
                    unpatched = [None] * len(destinations)
                for offset, continuation in enumerate(continued):
                    destination = destinations[offset]
                    source = sources[offset]
                    top = int(logits[offset].argmax())
                    record = {
                        "site": site, "plan": plan_name,
                        "source_id": records[source]["id"], "destination_id": records[destination]["id"],
                        "destination_state": records[destination]["state"],
                        "variant": records[source]["variant"] if plan_name != "reverse" else records[destination]["variant"],
                        "donor_class": records[source]["donor_class"] if plan_name != "reverse" else records[destination]["donor_class"],
                        "field": records[destination]["field"],
                        "source_is_primary": is_primary(records[source]) if plan_name != "reverse" else False,
                        "unpatched_margin": float(base_margin[destination]),
                        "patched_margin": float(margins[offset]),
                        "source_margin": float(base_margin[source]),
                        "margin_delta": float(margins[offset] - base_margin[destination]),
                        "patched_top5": top_record(logits[offset]),
                        "patched_top_in_null_family": top in null_family,
                        "patched_top_in_quote_family": top in quote_family,
                        "continuation": continuation,
                        "unpatched_continuation": unpatched[offset],
                    }
                    if plan_name == "matched":
                        record["kind"] = matched_meta[offset][0]
                    elif plan_name == "mismatched":
                        record["kind"] = mismatched_meta[offset][0]
                        record["mismatch_source_id"] = mismatched_meta[offset][3]
                    else:
                        record["kind"] = "reverse_absent_into_donor"
                    transfers.append(record)
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"transfer {site} {plan_name} {len(destinations)}", flush=True)
            print(f"{site} mismatched controls unavailable {unavailable_mismatch}", flush=True)

        own = patched_logits(list(range(len(records))), list(range(len(records))), PRIMARY_SITE)
        self_error = float((own - base_logits).abs().max())
        rolled = [(index + 1) % len(records) for index in range(len(records))]
        residual = patched_logits(list(range(len(records))), rolled, RESIDUAL_SITE)
        residual_error = float((residual - base_logits[rolled]).abs().max())
        print(f"self-patch max logit diff {self_error}; residual donor-logit max diff {residual_error}", flush=True)

    after = digest_parameters(model)

    def pack(rows):
        return {
            "cases": len(rows),
            "target_correct": sum(row["target_correct"] for row in rows),
            "target_is_null": sum(row["target_is_null"] for row in rows),
            "retained_fields_correct": sum(row["retained_fields_correct"] for row in rows),
            "schema_valid": sum(row["schema_valid"] for row in rows),
            "missing_required_cases": sum(bool(row["missing_required"]) for row in rows),
        }

    def transfer_pack(rows):
        paired = [row for row in rows if row["continuation"] and row["unpatched_continuation"]]
        absent = [row for row in paired if row["destination_state"] == "absent"]
        present = [row for row in paired if row["destination_state"] == "present"]
        return {
            "cases": len(rows),
            "mean_margin_delta": (sum(row["margin_delta"] for row in rows) / len(rows)) if rows else None,
            "patched_margin_prefers_null": sum(row["patched_margin"] > 0 for row in rows),
            "margin_flips_to_null": sum(row["unpatched_margin"] < 0 < row["patched_margin"] for row in rows),
            "patched_top_in_null_family": sum(row["patched_top_in_null_family"] for row in rows),
            "patched_continuation": pack([row["continuation"] for row in paired]) if paired else None,
            "same_batch_unpatched_continuation": pack([row["unpatched_continuation"] for row in paired]) if paired else None,
            "absent_cases": len(absent),
            "absent_repaired": sum(
                (not row["unpatched_continuation"]["target_correct"]) and row["continuation"]["target_correct"]
                for row in absent
            ),
            "present_cases": len(present),
            "present_value_preserved": sum(row["continuation"]["target_correct"] for row in present),
            "present_value_erased_to_null": sum(row["continuation"]["target_is_null"] for row in present),
        }

    generation_groups = defaultdict(list)
    for row in generations:
        generation_groups[(row["donor_class"], row["variant"], row["state"])].append(row)
    prefix_summary = defaultdict(list)
    for record in records:
        prefix_summary[f"{record['donor_class']}:{record['variant']}:{record['state']}"].append(decision_by_id[record["id"]])
    transfer_summary = defaultdict(list)
    for row in transfers:
        transfer_summary[(row["donor_class"], row["variant"], row["kind"], row["site"], bool(row["source_is_primary"]))].append(row)

    original_match = None
    saved = ROOT / "experiments/null_pairs/activation-probe-v2/generations.jsonl"
    if saved.exists():
        previous = {json.loads(line)["id"]: json.loads(line)["raw"] for line in saved.read_text().splitlines()}
        current = {row["id"]: row["raw"] for row in generations if row["variant"] == "original"}
        original_match = sum(current[key] == previous[key] for key in current)

    summary = {
        "model_revision": MODEL_REVISION,
        "run": "donor-transfer-v1",
        "records": len(records),
        "donors": 120,
        "batch_size": BATCH,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "modified_parameters": 0,
        "self_patch_max_logit_difference": self_error,
        "residual_swap_max_logit_difference": residual_error,
        "null_token_family": null_family,
        "quote_token_family": quote_family,
        "original_generation_exact_matches_to_v2": original_match,
        "free_generation": {
            f"{donor_class}:{variant}:{state}": pack(rows)
            for (donor_class, variant, state), rows in sorted(generation_groups.items())
        },
        "free_generation_success_ids": [record["id"] for record in donors],
        "primary_donor_ids": [record["id"] for record in donors if is_primary(record)],
        "prefix": {
            key: {
                "cases": len(rows),
                "top_in_null_family": sum(row["top_in_null_family"] for row in rows),
                "margin_positive": sum(row["null_minus_quote"] > 0 for row in rows),
                "continuation_target_correct": sum(row["target_correct"] for row in rows),
                "continuation_target_null": sum(row["target_is_null"] for row in rows),
                "continuation_retained": sum(row["retained_fields_correct"] for row in rows),
                "continuation_schema_valid": sum(row["schema_valid"] for row in rows),
            }
            for key, rows in sorted(prefix_summary.items())
        },
        "transfers": {
            f"{'primary' if primary else 'other-success'}:{donor_class}:{variant}:{kind}:{site}": transfer_pack(rows)
            for (donor_class, variant, kind, site, primary), rows in sorted(transfer_summary.items())
        },
        "elapsed_seconds": time.perf_counter() - started,
        "caveat": "Synthetic probes only. Prefix continuations are not free generations from the original prompt. Transfer continuations are compared with unpatched continuations from the same destination batches. literal_null copying is not evidence that the model represents missing evidence. No weight patch is selected by this script.",
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"case_ids": [record["id"] for record in records],
                "activations": {name: value.cpu() for name, value in activations.items()}}, RUN_DIR / "activations.pt")
    print(json.dumps({
        "free_generation_successes": len(donors),
        "primary_donors": len(summary["primary_donor_ids"]),
        "elapsed_seconds": summary["elapsed_seconds"],
        "parameter_sha256": after,
        "self_patch_max_logit_difference": self_error,
        "residual_swap_max_logit_difference": residual_error,
    }, indent=2), flush=True)
    if before != after:
        raise AssertionError("parameters changed")
    if self_error != 0 or residual_error != 0:
        raise AssertionError(f"control failed self={self_error} residual={residual_error}")


if __name__ == "__main__":
    main()

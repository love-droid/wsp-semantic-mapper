"""Controlled synthetic null diagnostics and activation swaps; never edits weights."""
import hashlib
import json
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_fieldbench_start import digest_parameters
from rescore_baseline import parse_fenced_json
from run_fieldbench import ROOT, MODEL_REVISION, build_prompt


def make_cases():
    cases = []
    specs = [
        ("insured_name", ["Named insured", "Policyholder name", "Name of proposer", "Primary insured"],
         lambda i: f"Cedar Workshop {i} Ltd", lambda i: f"Broker: Morgan Reed {i}\nClaims contact: Taylor Lee {i}"),
        ("policy_number", ["Policy number", "Policy no.", "Policy identifier", "Contract number"],
         lambda i: f"POL-740{i}", lambda i: f"Quote reference: QTE-931{i}\nPrevious policy number: OLD-830{i}"),
        ("loss_date", ["Date of loss", "Incident date", "Loss occurred on", "Date of incident"],
         lambda i: f"2025-03-{10+i:02d}", lambda i: f"Form expiration date: 2027-09-{10+i:02d}\nReport prepared: 2025-04-{10+i:02d}"),
    ]
    for field, labels, value_fn, distractor_fn in specs:
        for i in range(8):
            value = value_fn(i)
            lines = ["Insurance record", f"Record ID: REC-{field}-{i}", "Insurer: Atlas Mutual",
                     f"{labels[i % 4]}: {value}", distractor_fn(i)]
            if i >= 4:
                lines = [lines[0], lines[4], lines[1], lines[3], lines[2]]
            target_line = f"{labels[i % 4]}: {value}"
            schema = {"type": "object", "additionalProperties": False,
                      "required": [field, "record_id", "insurer"],
                      "properties": {field: {"type": ["string", "null"], "description": "Current policy number only; exclude quotes and previous policies." if field == "policy_number" else "Extract only the explicitly supported value for this role."},
                                     "record_id": {"type": "string"}, "insurer": {"type": "string"}}}
            for state in ("present", "absent"):
                text = "\n".join(lines if state == "present" else [line for line in lines if line != target_line])
                cases.append({"id": f"{field}-{i}-{state}", "pair_id": f"{field}-{i}", "state": state,
                              "field": field, "input": text, "json_schema": schema,
                              "expected": {field: value if state == "present" else None,
                                           "record_id": f"REC-{field}-{i}", "insurer": "Atlas Mutual"}})
            assert cases[-2]["input"].replace(target_line + "\n", "") == cases[-1]["input"]
    return cases


def main():
    out = ROOT / "experiments/null_pairs/activation-probe-v2"
    out.mkdir(parents=True, exist_ok=False)
    cases = make_cases()
    (out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.requires_grad_(False)
    torch.set_num_threads(8)
    before = digest_parameters(model)
    null = tok.encode("null", add_special_tokens=False)
    quote = tok.encode('"', add_special_tokens=False)
    assert len(null) == len(quote) == 1
    null_id, quote_id = null[0], quote[0]
    decoded = [tok.decode([i]).lstrip() for i in range(len(tok))]
    null_family = [i for i,s in enumerate(decoded) if re.match(r'^null(?:$|[\s,}])',s)]
    quote_family = [i for i,s in enumerate(decoded) if s.startswith('"')]
    def decision_margin(logits):
        return torch.logsumexp(logits[:,null_family],dim=-1)-torch.logsumexp(logits[:,quote_family],dim=-1)
    batch_size = 16
    for c in cases:
        messages, _ = build_prompt(c)
        c["prompt_ids"] = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        # Identical assistant prefix for both members of a pair, ending at value position.
        c["decision_ids"] = c["prompt_ids"] + tok.encode('{"' + c["field"] + '":', add_special_tokens=False)

    def encode(batch, key):
        return tok.pad([{"input_ids": c[key], "attention_mask": [1]*len(c[key])} for c in batch], padding=True, return_tensors="pt").to("cuda")

    def forward(batch):
        x = encode(batch, "decision_ids")
        positions = x["attention_mask"].long().cumsum(-1)-1
        positions.masked_fill_(x["attention_mask"] == 0, 0)
        return model(**x, position_ids=positions, use_cache=False, logits_to_keep=1).logits[:, -1].float()

    sites = {}
    for i, layer in enumerate(model.model.layers):
        sites[f"layer{i}.ffn"] = layer.feed_forward
        sites[f"layer{i}.attention" if layer.is_attention_layer else f"layer{i}.conv"] = layer.self_attn if layer.is_attention_layer else layer.conv
    sites["layer15.residual_control"] = model.model.layers[15]
    activations = defaultdict(list)
    hooks = []
    for name, module in sites.items():
        def capture(module, args, output, name=name):
            value = output[0] if isinstance(output, tuple) else output
            activations[name].append(value[:, -1].detach().clone())
        hooks.append(module.register_forward_hook(capture))
    started = time.perf_counter()
    with torch.inference_mode():
        base_logits = torch.cat([forward(cases[s:s+batch_size]) for s in range(0, len(cases), batch_size)])
    for h in hooks:
        h.remove()
    activations = {k: torch.cat(v) for k, v in activations.items()}
    base_margin = decision_margin(base_logits)
    base = []
    for i, c in enumerate(cases):
        values, ids = base_logits[i].topk(5)
        base.append({"id": c["id"], "null_minus_quote": float(base_margin[i]),
                     "top5": [{"id": j, "text": tok.decode([j]), "logit": v} for j,v in zip(ids.tolist(),values.tolist())]})
    (out / "decision_baseline.json").write_text(json.dumps(base, indent=2))

    results = []
    with torch.inference_mode(), (out / "activation_swaps.jsonl").open("w") as stream:
        for name, module in sites.items():
            changed = []
            for start in range(0, len(cases), batch_size):
                batch = cases[start:start+batch_size]
                donor = torch.tensor([i ^ 1 for i in range(start,start+len(batch))], device="cuda")
                replacement = activations[name][donor]
                def patch(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    value = value.clone()
                    value[:, -1] = replacement
                    return (value, *output[1:]) if isinstance(output, tuple) else value
                hook = module.register_forward_hook(patch)
                try:
                    logits = forward(batch)
                    changed.append(decision_margin(logits))
                finally:
                    hook.remove()
            margin = torch.cat(changed)
            delta = margin-base_margin
            record = {"site": name, "mean_absent_to_present_null_margin_change": float(delta[0::2].mean()),
                      "mean_present_to_absent_null_margin_change": float(delta[1::2].mean()),
                      "absent_quote_to_null_flips": int(((base_margin[1::2]<0)&(margin[1::2]>0)).sum()),
                      "present_null_to_quote_flips": int(((base_margin[0::2]>0)&(margin[0::2]<0)).sum()),
                      "per_case_patched_margin": margin.tolist()}
            results.append(record)
            stream.write(json.dumps(record)+"\n"); stream.flush()
            print(name, round(float(delta[0::2].mean()),3), round(float(delta[1::2].mean()),3), flush=True)

    # Self-patch control at the strongest non-residual site: should reproduce baseline.
    best = max([r for r in results if "control" not in r["site"]], key=lambda r: abs(r["mean_absent_to_present_null_margin_change"])+abs(r["mean_present_to_absent_null_margin_change"]))
    own = []
    with torch.inference_mode():
        for start in range(0,len(cases),batch_size):
            replacement = activations[best["site"]][start:start+batch_size]
            def self_patch(module,args,output):
                value = output[0] if isinstance(output,tuple) else output
                value = value.clone(); value[:,-1] = replacement
                return (value,*output[1:]) if isinstance(output,tuple) else value
            hook = sites[best["site"]].register_forward_hook(self_patch)
            try:
                own.append(forward(cases[start:start+batch_size]))
            finally:
                hook.remove()
    self_error = float((torch.cat(own)-base_logits).abs().max())
    assert self_error == 0, f"Self-patch changed logits: {self_error}"

    # Full unprefilled generation measures whether the constructed cases actually fail.
    generations = []
    with torch.inference_mode(), (out / "generations.jsonl").open("w") as stream:
        for start in range(0,len(cases),batch_size):
            batch = cases[start:start+batch_size]
            x = encode(batch,"prompt_ids"); width=x["input_ids"].shape[1]
            seq = model.generate(**x, do_sample=False, max_new_tokens=256, repetition_penalty=1.0, use_cache=True)
            for c, tokens in zip(batch,seq):
                raw=tok.decode(tokens[width:],skip_special_tokens=True)
                parsed,error,fence=parse_fenced_json(raw)
                is_object=isinstance(parsed,dict)
                record={"id":c["id"],"state":c["state"],"raw":raw,"parsed":parsed,"parse_error":error,
                        "target_key_present":is_object and c["field"] in parsed,
                        "target_correct":is_object and c["field"] in parsed and parsed[c["field"]]==c["expected"][c["field"]],
                        "retained_fields_correct":is_object and all(parsed.get(k)==c["expected"][k] for k in ("record_id","insurer"))}
                generations.append(record);stream.write(json.dumps(record)+"\n")
            stream.flush();print(f"Full generations {start+len(batch)}/{len(cases)}",flush=True)
    after=digest_parameters(model)
    assert before==after
    summary={"model_revision":MODEL_REVISION,"pairs":24,"cases":48,"synthetic_development_only":True,
             "batch_size":batch_size,"dtype":"bfloat16","torch_version":torch.__version__,"transformers_version":__import__("transformers").__version__,
             "parameter_sha256_before":before,"parameter_sha256_after":after,"modified_parameters":0,
             "null_token_id":null_id,"quote_token_id":quote_id,"self_patch_max_logit_difference":self_error,
             "null_token_family":null_family,"quote_token_family":quote_family,
             "present_prefers_null_over_quote":int((base_margin[0::2]>0).sum()),
             "absent_prefers_null_over_quote":int((base_margin[1::2]>0).sum()),
             "free_generation":{state:{"target_correct":sum(r["target_correct"] for r in generations if r["state"]==state),
                                        "retained_fields_correct":sum(r["retained_fields_correct"] for r in generations if r["state"]==state)} for state in ("present","absent")},
             "strongest_site_by_absolute_bidirectional_effect":best,"elapsed_seconds":time.perf_counter()-started,
             "caveat":"Teacher-forced prefix and null-versus-quoted-string token-family log-mass margin isolate one decision, not full value probability. Partial spellings of null are excluded. Final-position activation swaps are causal interventions on activations, not evidence that modifying the associated weights is safe or effective. Synthetic probes are not held-out evaluation."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    torch.save({"case_ids":[c["id"] for c in cases],"activations":{k:v.cpu() for k,v in activations.items()}},out/"activations.pt")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()

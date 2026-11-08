"""Read-only first-token diagnostic on the existing development partition."""

import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_fieldbench import ROOT, DATA_DIR, MODEL_REVISION, build_prompt


def digest_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    checkpoint = ROOT / "models/LFM2.5-350M-9921cd4d"
    source = DATA_DIR / "development.jsonl"
    baseline_dir = ROOT / "experiments/fieldbench/fieldbench-dev-original-v2"
    output = ROOT / "experiments/fieldbench/start-token-probe-v1"
    output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    baseline = {item["id"]: item for item in map(json.loads, (baseline_dir / "predictions.jsonl").read_text().splitlines())}
    original_info = json.loads((baseline_dir / "run_info.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_info["input_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    model.requires_grad_(False)
    before = digest_parameters(model)
    fence_ids = tokenizer.encode("```", add_special_tokens=False)
    brace_ids = tokenizer.encode("{", add_special_tokens=False)
    assert len(fence_ids) == len(brace_ids) == 1
    fence, brace = fence_ids[0], brace_ids[0]
    captured = []

    def capture_hidden(module, args):
        captured.append(args[0][0, -1].detach().float().cpu())

    hook = model.get_output_embeddings().register_forward_pre_hook(capture_hidden)
    records, hidden_states = [], []
    with (output / "cases.jsonl").open("w") as stream, torch.inference_mode():
        for index, case in enumerate(cases, 1):
            messages, prompt_hash = build_prompt(case)
            assert prompt_hash == baseline[case["id"]]["prompt_sha256"]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            captured.clear()
            result = model.generate(**inputs, do_sample=False, max_new_tokens=1, repetition_penalty=1.0, use_cache=True, return_dict_in_generate=True, output_scores=True)
            logits = result.scores[0][0].float()
            values, indices = logits.topk(5)
            generated_id = int(result.sequences[0, -1])
            baseline_first = tokenizer.encode(baseline[case["id"]]["raw_output"], add_special_tokens=False)[0]
            record = {
                "id": case["id"], "category": case["category"], "source": case["source"],
                "prompt_sha256": prompt_hash, "generated_token_id": generated_id,
                "matches_baseline_first_token": generated_id == baseline_first,
                "fence_logit": float(logits[fence]), "brace_logit": float(logits[brace]),
                "fence_minus_brace": float(logits[fence] - logits[brace]),
                "brace_rank": int((logits > logits[brace]).sum()) + 1,
                "top5": [{"id": int(i), "text": tokenizer.decode([i]), "score": float(v)} for i, v in zip(indices.tolist(), values.tolist())],
            }
            assert len(captured) == 1
            hidden_states.append(captured[0])
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if index % 20 == 0:
                print(f"Probed {index}/{len(cases)}", flush=True)
    hook.remove()
    after = digest_parameters(model)
    assert before == after, "Parameter integrity check failed"
    torch.save({"case_ids": [r["id"] for r in records], "final_hidden_states": torch.stack(hidden_states)}, output / "activations.pt")
    margins = [r["fence_minus_brace"] for r in records]
    summary = {
        "case_count": len(records), "split": "development", "model_revision": MODEL_REVISION,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "modified_parameters": 0, "gradient_updates": 0,
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(), "dtype": "bfloat16",
        "tied_embeddings": model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
        "fence_token_id": fence, "brace_token_id": brace,
        "fence_first_count": sum(r["generated_token_id"] == fence for r in records),
        "baseline_first_token_matches": sum(r["matches_baseline_first_token"] for r in records),
        "brace_rank_counts": dict(Counter(r["brace_rank"] for r in records)),
        "fence_minus_brace": {"min": min(margins), "median": statistics.median(margins), "mean": statistics.mean(margins), "max": max(margins)},
        "interpretation": "Observational diagnostic, not a causal intervention or evidence of improved extraction. Held-out partition not accessed.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

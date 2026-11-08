"""Print the architecture config and tensor inventory for LFM2.5-350M."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"

# Keep this model's cache inside the project. It is ignored by Git.
os.environ.setdefault("HF_HOME", str(ROOT / "models/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models/.cache/huggingface/hub"))

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


def inspect_model() -> dict:
    snapshot = Path(snapshot_download(
        MODEL_ID,
        revision=REVISION,
        cache_dir=os.environ["HF_HUB_CACHE"],
    ))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    weights_path = snapshot / "model.safetensors"

    tensors = []
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        for name in weights.keys():
            tensor = weights.get_slice(name)
            tensors.append({
                "name": name,
                "shape": tensor.get_shape(),
                "dtype": tensor.get_dtype(),
                "parameters": tensor.get_shape() and _product(tensor.get_shape()),
            })

    architecture = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "architecture": getattr(config, "architectures", None),
        "model_type": config.model_type,
        "parameter_count": sum(t["parameters"] for t in tensors),
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocabulary_size": getattr(config, "vocab_size", None),
        "layer_count": getattr(config, "num_hidden_layers", None),
        "layer_types": getattr(config, "layer_types", None),
        "attention_heads": getattr(config, "num_attention_heads", None),
        "key_value_heads": getattr(config, "num_key_value_heads", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "tie_word_embeddings": getattr(config, "tie_embedding", getattr(config, "tie_word_embeddings", None)),
        "weight_file": str(weights_path),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    return architecture


def _product(shape: list[int]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


if __name__ == "__main__":
    report = inspect_model()
    output_dir = ROOT / "experiments/model_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "architecture.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "tensors"}, indent=2) + "\n"
    )
    (output_dir / "parameter_names.txt").write_text(
        "\n".join(
            f"{tensor['name']}\t{tuple(tensor['shape'])}\t{tensor['dtype']}"
            for tensor in report["tensors"]
        ) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "tensors"}, indent=2))
    print(f"Parameter inventory: {output_dir / 'parameter_names.txt'}")

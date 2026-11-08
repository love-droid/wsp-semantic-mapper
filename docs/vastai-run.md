# Run the baseline on Vast.ai

This guide runs frozen-model inference only. It does not train the model or edit weights. Do not upload insurance PDFs or predictions to a cloud instance unless you have decided that cloud processing is appropriate; the starter dataset and outputs contain private insurance information.

## 1. Build the runner image locally

From the repository root, use Docker to build the CUDA runtime image:

```bash
docker build -f Dockerfile.vast -t wsp-semantic-mapper:vast .
```

Push it to a container registry you control, then configure a Vast.ai instance to use that image. Choose a GPU with at least 12 GB VRAM for this 350M model and leave enough disk for the roughly 1 GB checkpoint plus Python packages and cache. Use an instance with secure SSH access and shut it down after retrieving outputs.

## 2. Copy the reviewed dataset to the instance

Do not copy the complete local `data/` directory. Prepare only the reviewed JSONL cases you have chosen to run, then transfer that file over SSH/SCP to `/workspace/wsp-semantic-mapper/data/local/development.seed.jsonl`. The Docker image intentionally excludes local documents, model files, experiment outputs, and credentials.

## 3. Start the container and fetch the pinned model

Attach persistent storage mounted at `/workspace/hf-cache`, and start the image with the repository directory at `/workspace/wsp-semantic-mapper`. In the container terminal:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="LiquidAI/LFM2.5-350M",
    revision="9921cd4d82c0e3986887aa4b376f9446729ef8a3",
    local_dir="/workspace/hf-cache/LFM2.5-350M-9921cd4d",
)
PY
```

The download should contain the model configuration, tokenizer files, and `model.safetensors`.

## 4. Run inference and retrieve results

Transfer the selected JSONL into `data/local/development.seed.jsonl`, then run:

```bash
python scripts/run_baseline.py \
  --model-path /workspace/hf-cache/LFM2.5-350M-9921cd4d \
  --device cuda --dtype auto --max-new-tokens 1000
```

Results are written under `experiments/baseline/<run-id>/`. Copy that run directory back to your local machine over SCP before terminating the instance. Results can contain sensitive extracted values; keep them private and do not push them to Git.

## Credentials

Authenticate to Vast.ai using its normal local CLI or web console. Never place API keys in this repository, Docker image, command history, or uploaded dataset. This repository does not need a Vast.ai API key inside the container to run inference.

## Comparability

The original checkpoint is BF16. CUDA `--dtype auto` uses BF16 when the GPU supports it, otherwise FP16. Record the actual GPU/dtype in `run_info.json`. Compare a weight edit against its baseline using the same GPU type, dtype, prompt, decoding settings, and evaluation cases; a CUDA result does not directly quantify latency against the earlier CPU run. For exact GPU comparisons, retain and reuse the same Vast.ai GPU type and dtype.

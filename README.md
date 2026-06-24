# Question Rewriter Fine Tuning

This repo fine-tunes a configurable base model with Unsloth so the model can
rewrite a prior question/answer conversation into one standalone follow-up
question. The default base model in [`fine_tuning/config.py`](fine_tuning/config.py)
is `Qwen3 0.6B`.

## Training data

Populate [`data/training-questions.json`](data/training-questions.json) as a JSON
array using this exact example shape:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "when did we last service the gate"
      },
      {
        "role": "assistant",
        "content": "The gate was last serviced on 2026-06-18 by Garage & Gate Experts."
      },
      {
        "role": "user",
        "content": "What did they do"
      }
    ],
    "rewrite": "What work was completed when Garage & Gate Experts serviced the gate on 2026-06-18?"
  }
]
```

## Run training

Training settings live in [`fine_tuning/config.py`](fine_tuning/config.py). Edit
that config object when you want to change paths, model settings, or LoRA
hyperparameters. The Ollama/GGUF quantization setting is controlled by
`ollama_gguf_quantization`.

Use the helper script as the supported workflow:

```powershell
.\run-training.ps1
```

By default, `run-training.ps1` also runs `ollama create` after a successful
foreground training run so the latest fine-tuned Ollama artifact is created
automatically in a Docker-only Ollama runtime. The repo now includes an
`ollama` service in `docker-compose.yml`, pinned to `ollama/ollama:0.30.10`,
with host port `11436` mapped to container port `11434` to avoid collisions
with other Ollama instances on the machine. In foreground mode the script exits
when the `fine-tuning` container finishes instead of leaving the Docker attach
session open. It now reuses the existing Docker image by default so repeated
runs do not keep creating large replacement images; pass `-Build` only when you
need to rebuild the image. The script uses its explicit `OutputModelName` /
`OllamaModelName` parameters rather than parsing model names from config. Use
`-SkipOllamaCreate` to skip the Ollama step, or `-Detach` if you only want to
start training and return immediately.

Artifacts are written under `outputs/<output_model_prefix>-<base_model_slug>/`.
With the default config, this is `outputs/question-rewriter-qwen3-0.6b/`.

## Ollama output

The training script attempts to export:

- LoRA adapter weights
- A merged 16-bit model
- A quantized GGUF model for Ollama
- A `Modelfile`

The Docker setup now persists Unsloth's `llama.cpp` install in a named volume, so
the expensive clone/build step used during GGUF export is reused across runs.
The actual model conversion and quantization still rerun when the fine-tuned
weights change.

The Ollama artifact is exported as a quantized GGUF model using the
`ollama_gguf_quantization` setting in [`fine_tuning/config.py`](fine_tuning/config.py).
By default this is `Q4_K_M`, so the fine-tuned Ollama model is exported in a
4-bit quantized format.

`run-training.ps1` is also the supported path for creating the Ollama model.
Its generated `Modelfile` points at the exported fine-tuned quantized `.gguf`
file in the same output directory.

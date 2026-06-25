# Demo

This folder contains a small Ollama chat demo for the fine-tuned rewrite model.

## Run

After `.\run-training.ps1` finishes and creates the Ollama model, run:

```powershell
.\run-demo.ps1
```

This starts the compose-managed `ollama` service if needed and runs the demo in
Docker against the Docker-hosted Ollama model runtime.

## What it shows

The demo prints one multi-turn homeowner conversation and asks the fine-tuned
model to rewrite it into a standalone question.

Use `.\run-demo.ps1 -Build` if the runtime image needs to be rebuilt after
dependency or Dockerfile changes.

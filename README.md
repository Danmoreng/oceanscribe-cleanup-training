# OceanScribe Cleanup Training

Small, local, reproducible fine-tuning pipeline for turning raw ASR transcripts
into clean text for OceanScribe.

This is intentionally a separate repository from the desktop application. The
training contract defines the model we want; OceanScribe will be adapted to the
selected model and context length after model quality has been demonstrated.

## Model-facing format

The payload is plain text. There is no JSON response and no long system prompt.

```text
Cleanup
Language: de-DE
Commands: on
<transcript>
äh hallo Peter neuer Absatz danke für deine Nachricht
</transcript>
Output:
```

The assistant completion is only:

```text
Hallo Peter.

Danke für deine Nachricht.
```

The formatter places these user and assistant messages into the official,
pinned Qwen chat template. At inference time generation stops at the model's
normal end-of-turn token.

## Repository layout

```text
configs/                   Training and dataset mixes
src/oceanscribe_cleanup/   Shared Python implementation
tests/                     Fast correctness tests
PLAN.md                    Milestones and decisions
```

Large or private material is local-only under ignored `data/`, `runs/`, and
`artifacts/` directories.

## Initial training plan

- Smoke: 400 examples, LoRA rank 16, 20-50 steps, 2,048 total tokens.
- v0: about 20,000 examples balanced approximately 50/50 between German and
  English by tokens; compare ranks 16, 32, and 64.
- Stage B: continue the best adapter on long examples at 4,096 total tokens.
- Full fine-tuning is not part of the initial plan.

The deployable artifact will eventually be a merged and quantized GGUF, but
application integration is deliberately outside the first repository setup.

## Target environment

Python 3.11 and `uv` are the supported setup. Linux or WSL2 is the preferred
training environment.

```bash
uv sync --group dev --group train
PYTHONPATH=src uv run python -m unittest discover -s tests
```

`uv.lock` is intentionally created after the first successful Qwen model-load
and mini-training smoke test on the target GPU. This avoids freezing an
untested Torch/CUDA combination on the laptop.

## Current status

The repository currently contains the agreed prompt formatter, initial configs,
tests, and a concise implementation plan. Dataset ingestion, model loading, the
collator, and training loop are the next milestone.

## License

The training code is licensed under the Apache License 2.0. Datasets, base
models, trained weights, and other external artifacts retain their own licenses
and must be documented separately.

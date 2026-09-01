# OceanScribe Cleanup Training

Small, local, reproducible fine-tuning infrastructure for turning raw ASR
transcripts into clean text for OceanScribe. This repository is deliberately
separate from the desktop application and contains no cloud inference or
service integration.

## Model-facing contract

`raw-v1` is the only implemented contract. Its literals and field order live in
`src/oceanscribe_cleanup/prompt_contracts/raw_v1.py`; run configs may select the
version but cannot override separators or tags.

```text
Cleanup
Language: de-DE
Commands: enabled
<terminology>
OceanScribe
Qwen3.5
</terminology>
<transcript>
äh hallo Peter neuer Absatz danke für deine Nachricht
</transcript>
Output:
```

The supervised completion contains only the cleaned transcript:

```text
Hallo Peter.

Danke für deine Nachricht.
```

There is no JSON, role message, system prompt, chat template, or newly added
special token. The training path tokenizes the prompt and target separately
with `add_special_tokens=False`, appends `tokenizer.eos_token_id` exactly once,
and supervises only target tokens plus that final EOS. Prompt and padding labels
are `-100`. Padding is masked by position, not by token value, because PAD and
EOS may share an ID.

Input validation currently accepts only `de-DE`, `en-US`, and `en-GB`, requires
a real Boolean for command handling, normalizes line endings, freezes
terminology as a tuple, and rejects structural delimiters or model-special-token
strings in payload fields. The transcript must contain dictated content; the
cleaned output may be empty.

The terminology block is always present. Terms are canonical spelling hints,
not words that must be inserted. Aliases are not part of `raw-v1`.

`raw-v1` is the primary product contract. Before it is declared permanently
frozen, a small research-only smoke comparison may measure it against the
official Qwen template using isolated artifacts and identical data. That
comparison does not alter the raw-v1 training or inference path.

## Repository layout

```text
configs/runs/             Concrete, single-rank run configurations
configs/sweeps/           Explicit matrices over concrete base configs
src/oceanscribe_cleanup/  Contracts, tokenization, collator, manifests, CLI
tests/                    CPU-only contract tests
DATA_SOURCES.md           Source licensing and redistribution registry
PLAN.md                   Remaining milestones and decisions
```

Large or private material remains local under ignored `data/`, `models/`,
`runs/`, `cache/`, and `artifacts/` directories. Derivatives of one source must
remain in one split, and gold evaluation data must not enter training or teacher
generation.

## Setup and checks

Python 3.11 and `uv` are the supported environment. Linux or WSL2 is preferred
for training.

```bash
uv sync --locked --group dev
uv run --locked oceanscribe-cleanup validate-configs
uv run --locked ruff check .
uv run --locked pytest tests -m "not model_download"
uv build
```

The dependency lock was generated after the target-machine model-load and
mini-training smoke succeeded with PyTorch 2.13, CUDA 13, and an RTX 4070 Ti.
Keep setup, CI, and run commands locked.

Regular CI does not download models. Manual tokenizer preflight requires a
pinned tag or commit and records the resolved repository SHA:

```bash
uv sync --locked --group train
uv run --locked oceanscribe-cleanup tokenizer-preflight \
  --revision <pinned-tag-or-commit>
```

Full module inspection first tries `Qwen3_5ForCausalLM`. Only if that checkpoint
cannot use the text-only class does it try the multimodal class and classify
vision, projector, embedding, LM-head, MTP, and language-backbone parameters:

```bash
uv run --locked oceanscribe-cleanup inspect-model \
  --revision <pinned-tag-or-commit>
```

The inspection prints the explicit language-backbone `Linear` allowlist for
LoRA. Training integration must call the included trainable-parameter assertion
after adapter injection.

## External-data smoke training

After downloading the pinned Sotto and Aawaaz snapshots into their ignored
`data/raw/` directories, build the deterministic 400-record English subset:

```bash
uv run --locked oceanscribe-cleanup prepare-external-smoke \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --per-source 200 \
  --validation-per-source 20 \
  --seed 42
```

Run the two-step integration smoke first, then the 50-step run:

```bash
uv run --locked oceanscribe-cleanup train \
  --config configs/runs/smoke-r16-quick.yaml \
  --dataset data/processed/smoke-en-400/records.jsonl \
  --output-dir runs/qwen35-08b-smoke-r16-quick

uv run --locked oceanscribe-cleanup train \
  --config configs/runs/smoke-r16.yaml \
  --dataset data/processed/smoke-en-400/records.jsonl \
  --output-dir runs/qwen35-08b-smoke-r16
```

Training loads only `Qwen3_5ForCausalLM`, rejects dataset/config/hash mismatch,
tokenizes without chat templates or truncation, supervises completion plus EOS,
injects LoRA only into the inspected language `Linear` allowlist, evaluates on
the held-out validation split, saves local manifests, and verifies adapter
reload. The RTX 4070 Ti smoke uses micro-batch 4 with four accumulation steps;
the four longest records pass a backward/optimizer stress test below 10 GiB of
PyTorch-allocated VRAM.

## Config and run reproducibility

All YAML files have `schema_version: 1`, reject unknown fields, and use
`enabled`, `disabled`, or `mixed` rather than YAML's ambiguous `on`/`off`.
Concrete runs contain one LoRA rank. The long-context run names an adapter path,
expected base revision, and prompt contract instead of resolving an implicit
“best” run.

`model.revision: null` means “preflight only” and must be replaced before a real
run. A real run manifest rejects floating revisions and records the repository
commit, config hash, resolved model/tokenizer revisions, dataset revisions,
seeds, Python/library/CUDA versions, and GPU details.

## Training outline

- Smoke: 400 examples, rank 16, 50 steps, 2,048 total tokens.
- v0: roughly 20,000 examples; compare ranks 16, 32, and 64 with aligned alpha.
- Stage B: continue an explicitly selected adapter on real contiguous examples
  at 4,096 total tokens.
- Packing and truncation are disabled. Oversized records fail clearly.
- Full fine-tuning is outside the initial plan.

The deployment artifact will eventually be a merged, quantized GGUF. Base-model
and merged-adapter conversion parity must be tested with pinned revisions before
application integration.

## License

The training code is Apache-2.0. Datasets, base models, trained weights, and
other external artifacts retain their own licenses and must be registered in
`DATA_SOURCES.md` and generated manifests.

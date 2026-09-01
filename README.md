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
uv sync --group dev
uv run oceanscribe-cleanup validate-configs
uv run ruff check .
uv run pytest -m "not model_download"
uv build
```

Before GPU work, verify and document the target CUDA/PyTorch wheel source. Per
the repository reproducibility rule, `uv.lock` is intentionally not committed
until the model-load and mini-training smoke succeeds on that target. Commit the
resulting lock immediately afterward and switch CI and documented commands to
`uv sync --locked` / `uv run --locked`.

Regular CI does not download models. Manual tokenizer preflight requires a
pinned tag or commit and records the resolved repository SHA:

```bash
uv sync --group train
uv run oceanscribe-cleanup tokenizer-preflight \
  --revision <pinned-tag-or-commit>
```

Full module inspection first tries `Qwen3_5ForCausalLM`. Only if that checkpoint
cannot use the text-only class does it try the multimodal class and classify
vision, projector, embedding, LM-head, MTP, and language-backbone parameters:

```bash
uv run oceanscribe-cleanup inspect-model \
  --revision <pinned-tag-or-commit>
```

The inspection prints the explicit language-backbone `Linear` allowlist for
LoRA. Training integration must call the included trainable-parameter assertion
after adapter injection.

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

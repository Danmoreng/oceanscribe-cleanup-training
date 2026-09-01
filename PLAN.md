# Implementation plan

## Fixed decisions

- Base model: `Qwen/Qwen3.5-0.8B-Base`.
- The implemented task is `raw-v1`: one plain completion sequence, no JSON,
  roles, chat template, or new special tokens.
- The exact inference prefix ends with `Output:\n`.
- Prompt and target are tokenized separately; only target plus one tokenizer EOS
  are supervised.
- The terminology block is always present and contains canonical hints only.
- Stage A uses BF16 LoRA at at most 2,048 total tokens; Stage B continues an
  explicit selected adapter at at most 4,096.
- Packing and silent truncation remain disabled.

## Completed contract-hardening milestone

- Strict versioned run/sweep config schemas and recursive validation CLI.
- `enabled`/`disabled`/`mixed` command enum; no YAML `on`/`off` ambiguity.
- Single-source `raw-v1` renderer with payload collision protection.
- Frozen cleanup records with source, split, provenance, edit, generator, and
  lineage fields; empty supervised outputs are valid.
- Token-ID EOS construction, completion-only labels, position-based padding,
  and explicit oversize errors.
- Dataset/run manifest schemas and non-floating revision checks.
- Pinned-tokenizer and model-inspection commands; text-only model is tried first.
- CPU tests, Hypothesis coverage, lint/build checks, and manual model preflight.

## Remaining milestones

### 0. Target-machine and dependency proof

On Python 3.11 Linux/WSL2, record OS, GPU, VRAM, driver, Torch, CUDA, BF16
support, disk space, Transformers, and PEFT versions. Select and document the
CUDA-specific PyTorch wheel source. Run the CPU suite, perform the model-load
and mini-training smoke, commit the resulting `uv.lock`, and switch CI to
`--locked` in the same change.

### 1. Pinned Qwen preflight

Resolve and pin the exact model/tokenizer revision. Verify EOS and PAD IDs and
run the model inspector. Confirm whether `Qwen3_5ForCausalLM` loads this
checkpoint. If the multimodal fallback is required, verify the reported module
classification manually. Freeze vision, projector, embeddings, LM head, and
MTP; inject LoRA only into the printed language `Linear` allowlist and abort on
any unexpected trainable parameter.

Run a small, isolated A/B smoke between `raw-v1` and the official Qwen template
before declaring the production contract permanently frozen. This is a
measurement task, not permission to apply a chat template to raw-v1 data.

### 2. Base GGUF compatibility proof

Before training, convert the pinned base revision with a pinned llama.cpp
commit. Compare 20 fixed greedy prompts between Transformers and llama.cpp,
including empty-output and EOS behavior. Record both revisions and outputs.

### 3. Data ingestion and manifests

Only after licensing entries in `DATA_SOURCES.md` are complete, ingest 100 Sotto
and 100 conservative Aawaaz examples. Generate 100 deterministic German and 100
deterministic English examples. Split source families before augmentation, keep
derivatives together, deduplicate, route oversize records, generate dataset
manifests, and keep a must-pass evaluation set out of training.

### 4. Mini-training

Run rank 16 for 50 steps at 2,048 tokens with the tested collator. Record average
prompt/target lengths, non-padding tokens per optimizer step, and oversize
counts. Save a complete run manifest, reload the adapter, and inspect German and
English results.

### 5. Merged GGUF proof

Merge Base + LoRA to an HF model, convert that result to GGUF, and repeat the
same pinned parity tests. Direct LoRA-to-GGUF conversion is not required.

### 6. v0 comparison

Build about 20,000 examples balanced by language tokens and features:

- 6,000 Sotto English;
- 2,000 conservative Aawaaz English;
- 2,000 deterministic English;
- 10,000 deterministic German.

Keep at least 20% preserve/no-op and 10% hard negatives. Include relevant,
already-correct, absent, irrelevant, and empty terminology cases. Compare ranks
16, 32, and 64 with identical data, seeds, and evaluation.

### 7. Long context

Replace the placeholder adapter path/revision in the Stage B config with the
selected immutable checkpoint. Continue on real contiguous documents at 4,096
tokens; never simulate long context by packing unrelated records.

### 8. v1 data loop

Only after v0 is convincing, add real disfluency data, production transcripts,
more German source text, teacher-naturalized examples, and broader evaluation.
Full fine-tuning remains a separate manual decision.

## Acceptance criteria

- Python 3.11 CPU CI, config validation, lint, tests, wheel build, and
  installed-wheel import pass; CI becomes locked with the post-GPU-smoke lock.
- Training/inference raw-v1 prefixes are identical and contain no chat controls.
- Prompt/padding labels are masked; target and final EOS remain supervised even
  when PAD equals EOS or the cleaned target is empty.
- Malformed locales, non-Boolean commands, structural injection, unknown config
  fields, floating revisions, and oversized examples fail explicitly.
- No excluded model component is trainable.
- The smoke adapter saves, reloads, and improves representative examples.
- The merged smoke model passes pinned GGUF/llama.cpp parity checks.

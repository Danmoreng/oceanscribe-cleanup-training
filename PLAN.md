# Implementation plan

## Decisions already made

- Use `Qwen/Qwen3.5-0.8B-Base`.
- Train a plain-text cleanup task with no JSON input or output.
- Use the official Qwen chat template without adding special tokens.
- The assistant completion contains only the cleaned transcript.
- Start with BF16 LoRA. Compare ranks 16, 32, and 64 only after the smoke run.
- Train at 2,048 total tokens first, then continue the best adapter at 4,096.
- Adapt OceanScribe's worker and context settings after the model is selected.

## Milestones

### 0. Target-machine check

Record Python, operating system, GPU, VRAM, driver, Torch, CUDA, BF16 support,
free disk space, Transformers, and PEFT versions. Use Python 3.11 and WSL2/Linux
on the training desktop.

### 1. Model inspection

Load the pinned Qwen model and tokenizer/processor. Render one prompt with the
official chat template. Report language, vision, projector, embedding, LM-head,
and MTP parameter groups. Discover eligible language-backbone linear modules.

### 2. LoRA and collator correctness

Inject LoRA only into discovered language modules and abort if any unexpected
parameter is trainable. Implement completion-only labels and tests for Unicode,
paragraphs, EOS, padding, empty targets, and oversized examples.

### 3. Smoke data

Ingest 100 Sotto and 100 conservative Aawaaz examples. Generate 100 deterministic
German and 100 deterministic English examples. Keep a small, separate must-pass
set out of training.

### 4. Mini-training

Run LoRA rank 16 for 20-50 steps at 2,048 total tokens. Verify loss, reload the
adapter, and inspect German and English cleanup outputs. Pin dependencies and
model/data revisions only after this succeeds.

### 5. Early deployment proof

Merge the smoke adapter, convert it to GGUF, and verify that the pinned
llama.cpp runtime can load and generate with the same plain-text prompt. This is
a compatibility proof, not application integration.

### 6. v0 comparison

Build about 20,000 examples balanced by language tokens and features:

- 6,000 Sotto English;
- 2,000 conservative Aawaaz English;
- 2,000 deterministic English;
- 10,000 deterministic German.

Keep at least 20% preserve/no-op and at least 10% hard-negative examples.
Compare ranks 16, 32, and 64 using identical data, seed, and evaluation.

### 7. Long context

Continue the best v0 adapter on real long-form examples at 4,096 total tokens.
Do not simulate long-context coverage by packing unrelated short samples.

### 8. v1 and Nemotron loop

Only after v0 is convincing, add real disfluency data, Nemotron production
transcripts, more German source text, teacher-naturalized examples, and broader
evaluation. Full fine-tuning remains a separate manual decision.

## Acceptance criteria for the first implementation pass

- All unit tests pass.
- Prompt formatting is stable and contains no JSON.
- Prompt and padding tokens are excluded from loss.
- No vision or other excluded component is trainable.
- Oversized examples are reported rather than truncated.
- A small adapter trains, saves, reloads, and improves representative commands.
- The merged smoke model passes a GGUF/llama.cpp compatibility check.

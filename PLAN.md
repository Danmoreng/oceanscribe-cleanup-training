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
- Deployment uses the neighboring `qwen35-cpu` engine and its custom
  H128/Q4-G32-DOT4 `.q35h` weights. `qwen35x` BF16 GPU inference is an optional
  evaluation backend after comparison with the Transformers reference.

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

## Completed target-machine smoke milestone

- Pinned `Qwen/Qwen3.5-0.8B-Base` at
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and generated `uv.lock` only
  after the model/training proof succeeded.
- Verified PyTorch 2.13 CUDA 13 BF16 on an RTX 4070 Ti and loaded the text-only
  `Qwen3_5ForCausalLM` path.
- Built a deterministic 400-record English Sotto/Aawaaz subset with 360 train
  and 40 validation records, pinned sources, deduplication, token limits, and a
  local dataset manifest.
- Completed the rank-16 50-step LoRA run, adapter save/reload, and validation
  loss check. Micro-batch 4 with four accumulation steps delivered the best
  safe throughput; batch 8 and batch 2 without checkpointing exceeded 12 GB.

## Completed RTX 5080 laptop smoke milestone

- Reused the successful pinned environment on the RTX 5080 Laptop (16 GB).
- Froze 400 unchanged English `KEEP` examples from the local reviewed CSV,
  preserving source-family holdouts: 360 train / 40 validation.
- Completed 50 rank-16 BF16 LoRA steps and verified adapter reload. Validation
  loss fell from 0.54124 to 0.16467; greedy held-out WER was 0.18621, with
  11/40 whitespace-normalized exact outputs and EOS on all 40 outputs.
- Peak training allocation was about 9.4 GiB. The longest example was 690
  total tokens, so this does not establish full 2,048-token batch capacity.
- Meaning changes remain in some outputs. Add independent German/English
  checks for negation, self-correction, omitted clauses, unsupported hints and
  preserve behavior before using this adapter in OceanScribe.
- Generated 200 original bilingual diagnostic records and a separate 50-record
  synthetic challenge suite. The existing smoke passed annotated checks on
  23/25 English and 17/25 German cases; real independently reviewed gold is
  still needed. German canonical-term correction failed both opportunities.
- Verified token-balanced sampling, best-checkpoint selection and adapter
  continuation through two separate two-step GPU runs.
- A 2,048-token capacity probe passed at micro-batch 1 (~7.3 GiB allocated);
  micro-batch 2 was marginal, and 4 failed. Use batch 1 / accumulation 16 for
  full-length Stage A training.
- Native import proof: merge preserved all six HF completions; both engine
  tokenizers matched all six prompts. GPU BF16 output matched 4/6 and
  uncalibrated CPU text 3/6. Investigate semantic/format regressions and collect
  cleanup calibration before promoting a native backend or final artifact.

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

Keep `raw-v1` identical in training, Transformers evaluation and both native
engines. The Base checkpoint and existing EOS remain the production contract.

### 2. Native engine compatibility proof

Record the native engine commits and binary hashes. Compare prompt token IDs
and fixed greedy completions against Transformers, including German/English,
empty output, EOS, terminology and literal instructions within transcripts.
Use raw `--prompt-file`, never a chat input flag. The GPU engine supports
`--stop-token 248044`; the CPU CLI stops on the tokenizer's existing EOS.

The engines expect `model.language_model.*` tensors; the text-only HF model
exports `model.*`. Prepare a separate native-loader directory with an explicit
lossless key mapping and tied embedding/head assertion. Keep a standard HF
directory for reload and reference checks.

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

### 5. Merged native export proof

Merge Base + the selected LoRA into BF16 HF weights. Verify that merging itself
preserves completions; compare the native GPU BF16 backend before using it for
long evaluations. Then pack H128/Q4-G32-DOT4 with `qwen35_cpu_pack` and run the
same quality suite on the native CPU engine. Exact quantized output identity
is not assumed: record output drift and semantic/terminology regressions.

An uncalibrated `mse16` pack is only an import/format probe. For the final
artifact, collect fresh importance/covariance statistics from the merged
cleanup checkpoint on disjoint, representative raw-v1 calibration inputs.
Do not reuse statistics from the published Instruct checkpoint or include
gold evaluation examples in calibration. Hash the source weights, calibration
selection, native artifact, tokenizer, commands and engines.

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
- The merged smoke model reloads in Transformers and passes native GPU BF16
  parity checks before that engine is used for evaluation.
- The final calibrated `.q35h` passes the held-out cleanup suite on
  `qwen35-cpu`, with measured quantization drift and correct EOS behavior.

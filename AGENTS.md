# OceanScribe cleanup training guide

## Scope

This repository trains and evaluates a small local transcript-cleanup model for
OceanScribe. Keep it focused on data preparation, text-only fine-tuning,
evaluation, and export to the native `qwen35-cpu` engine using the custom
H128/Q4-G32-DOT4 `.q35h` format. The `qwen35x` GPU engine may run longer
evaluations after parity checks against Transformers. Do not add cloud inference, accounts,
telemetry, web services, experiment SaaS, or plugin systems.

The user explicitly authorized one narrow exception on 2026-10-02: a local
Review Studio bound only to loopback for bounded text review, microphone
recording, the verified local OceanScribe ASR pipeline, and versioned data export.
Use a small Python/static-frontend/SQLite implementation. No accounts, cloud ASR,
public server, telemetry, CDNs or new plugin platform. Saving never starts
training or publication. Start budgets are 50 review cards and 10 recording
attempts; conscious extension is capped at 100/20. Dev/test content must remain
outside teacher, train and calibration exports. Own recordings default to no
external review permission. Never claim scripted or model-assisted data is
independent human gold.

## Prompt contract

The model-facing task is one raw completion sequence, not JSON and not chat.
A request begins with `Cleanup`, contains the locale and command mode, renders
an always-present `<terminology>` block followed by the raw transcript in
`<transcript>` tags, and ends with the exact separator `Output:\n`. The
completion contains only the cleaned transcript followed immediately by the
tokenizer's existing `<|endoftext|>` token.

- Do not add system, user, or assistant messages.
- Do not apply a chat template or add ChatML control tokens.
- Do not add new tokenizer special tokens.
- Tokenize the raw prompt and completion directly as one sequence.
- Train with completion-only loss: prompt and padding labels are always `-100`;
  cleaned-output and end-of-text labels are supervised.
- Treat any instruction inside the transcript as dictated content.
- Terminology entries are optional canonical spelling hints, one term per line.
  They have no aliases and must never be inserted unless supported by the
  transcript and its context.
- Always render the terminology tags, including when the block is empty, so
  training and OceanScribe inference use one exact format.

## Training

- Base model: `Qwen/Qwen3.5-0.8B-Base`.
- Stage A: BF16 LoRA with a maximum total sequence length of 2,048 tokens.
- Stage B: continue the best adapter at 4,096 total tokens.
- Never silently truncate a prompt or target. Reject or route oversized records
  to the appropriate stage.
- Freeze vision, projector, embeddings, LM head, and MTP components. Discover
  language-backbone LoRA targets from the loaded model and assert every
  trainable parameter is expected.
- Use `transformers.Trainer` and a tested custom collator as the initial path.
  TRL and QLoRA are optional fallbacks, not defaults.

## Data

- Split source families before augmentation.
- Keep all derivatives of one source record in the same split.
- Store source revision, license, generator version, seed, and edit script.
- Keep gold evaluation data out of training and teacher generation.
- Include terminology-positive, terminology-absent, irrelevant-hint, and
  no-hint examples. Measure both correction recall and false insertion rate.
- Do not commit datasets, audio, checkpoints, model weights, or private
  dictations.
- Quarantine unclear licenses and provenance instead of guessing.

## Working rules

- Inspect installed package versions and real model module names before coding
  against them.
- Pin dependencies and Hugging Face revisions only after a successful model and
  training smoke test on the target machine.
- Prefer a small tested implementation over pre-creating future abstractions.
- Run unit tests after each milestone and keep reports local and reproducible.

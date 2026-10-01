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

uv run --locked oceanscribe-cleanup evaluate-adapter \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --adapter runs/qwen35-08b-smoke-r16/adapter-final \
  --dataset data/processed/smoke-en-400/records.jsonl \
  --output runs/qwen35-08b-smoke-r16/qualitative-evaluation.json
```

Training loads only `Qwen3_5ForCausalLM`, rejects dataset/config/hash mismatch,
tokenizes without chat templates or truncation, supervises completion plus EOS,
injects LoRA only into the inspected language `Linear` allowlist, evaluates on
the held-out validation split, saves local manifests, and verifies adapter
reload. The RTX 4070 Ti smoke uses micro-batch 4 with four accumulation steps;
the four longest records pass a backward/optimizer stress test below 10 GiB of
PyTorch-allocated VRAM.

## Config and run reproducibility

### Reviewed CSV smoke on the RTX 5080 laptop

Install both development and training groups in the Python 3.11 environment:

```bash
uv sync --locked --group dev --group train
uv run --locked pytest tests -m "not model_download"
```

The local review CSV can produce a frozen 400-example English smoke subset.
Only `KEEP` rows with unchanged original references and the registered upstream
revisions/licenses are eligible; Aawaaz also passes the conservative filter.
German translations and changed or uncertain references are excluded from this
technical smoke pending complete provenance review. A deterministic family split
selects 180 train and 20 validation records per source. `selection.json` records
the source CSV hash, exclusions, and family assignments; future translations of
these records must inherit those assignments. No examples are truncated.

```bash
uv run --locked oceanscribe-cleanup prepare-reviewed-smoke \
  --csv data/oceanscribe-review-en-de-intermediate-recovered-1175.csv \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --output-dir data/processed/reviewed-keep-en-400

uv run --locked oceanscribe-cleanup train \
  --config configs/runs/reviewed-keep-smoke-r16-quick.yaml \
  --dataset data/processed/reviewed-keep-en-400/records.jsonl \
  --output-dir runs/qwen35-08b-reviewed-keep-smoke-r16-quick

uv run --locked oceanscribe-cleanup train \
  --config configs/runs/reviewed-keep-smoke-r16.yaml \
  --dataset data/processed/reviewed-keep-en-400/records.jsonl \
  --output-dir runs/qwen35-08b-reviewed-keep-smoke-r16
```

Run directories must be new. The quick run takes two optimizer steps; the main
smoke takes 50 with rank 16, BF16, microbatch 4, and accumulation 4. Both use the
same fixed validation set. This no-hint, commands-enabled smoke verifies the
training path; it does not establish German or terminology quality.

The October 1, 2026 laptop smoke passed with PyTorch 2.13.0/CUDA 13.0,
Transformers 5.16.1, and PEFT 0.20.0. The 50-step training took 96 seconds,
used 9.4 GiB peak allocated VRAM, and reduced held-out loss from 0.541 to 0.165.
All 40 greedy validation generations stopped at EOS without prompt control
markers; 11 matched their references after whitespace normalization. Some
outputs still omit information, so this establishes technical viability rather
than production quality. The longest dataset record is 690 total tokens; a
full 2,048-token memory proof remains outstanding. Reports, source hashes,
family holdouts, and a code snapshot are saved locally under
`runs/preflight-rtx5080/` and the concrete run directory.

```bash
uv run --locked oceanscribe-cleanup evaluate-adapter \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --adapter runs/qwen35-08b-reviewed-keep-smoke-r16/adapter-final \
  --dataset data/processed/reviewed-keep-en-400/records.jsonl \
  --output runs/qwen35-08b-reviewed-keep-smoke-r16/qualitative-evaluation.json \
  --limit 40 --batch-size 4 --max-new-tokens 512
```

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

The deployment artifact is a merged model in the custom H128/Q4-G32-DOT4
`.q35h` format used by the neighboring `qwen35-cpu` engine. Longer evaluations
may use the neighboring `qwen35x` BF16 GPU engine after parity checks against
Transformers. Both receive the exact raw-v1 prompt, with no chat template.

Export the selected adapter into a standard HF directory and a separate
native-loader directory:

```bash
uv run oceanscribe-cleanup merge-adapter \
  --adapter runs/qwen35-08b-reviewed-keep-smoke-r16/adapter-final \
  --output-dir artifacts/reviewed-smoke-native-v1
```

The native directory translates `model.*` language tensor names into the
engines' `model.language_model.*` layout without changing tensor values. The
tied output head shares the embedding matrix. Its export manifest hashes the
adapter, parent run manifest and generated files; export alone does not prove
inference parity.

For GPU evaluation, use `--infer-gpu --hf-model-dir .../native --prompt-file ...
--stop-token 248044 --temperature 0 --repeat-penalty 1 --gpu-bf16
--gpu-decode-backend qwen35x --qwen35x-weight-precision bf16`. For CPU deployment, pack that same
native directory with `qwen35_cpu_pack`, then supply `--model-dir .../native
--weights .../model.q35h --prompt-file ...` to `qwen35_cpu`. An uncalibrated
`mse16` conversion is only a format probe. The final quantization requires fresh
importance/covariance calibration from the merged cleanup model, kept disjoint
from evaluation. Record semantic and terminology regressions from quantization;
the published Instruct artifact is not the trained Base + LoRA checkpoint.

The local six-case import probe succeeded on both engines. HF adapter and merged
HF completions matched in all six cases; both native prompt tokenizers also
matched HF. GPU BF16 completions matched in four cases and uncalibrated CPU text
in three. A licensee/licensor divergence changed meaning, and the CPU probe
failed the output format on filler-only input. These results establish import
support, not production quality or full parity. The local commands and outputs
are under `runs/native-engine-proof/`; final calibration remains pending.

## Bilingual preparation and capacity checks

Generate 200 original template records (160 train / 40 validation, equally split
between English and German) plus 50 separate synthetic challenge records:

```bash
uv run oceanscribe-cleanup prepare-targeted \
  --output-dir data/processed/targeted-bilingual-v1 --seed 42
```

Families are split before locale/hint variants. Training includes 20% explicitly
marked preserve examples and 20% hard negatives, relevant/absent/no hints and
both command modes. Each record retains the generator SHA-256, seed, template
family and edit script. The challenge set shares template logic with training;
it is a diagnostic set, not independent human gold. It remains outside training
and teacher generation.

`targeted-bilingual-best-quick.yaml` and `targeted-bilingual-continue-quick.yaml`
successfully exercised language-token balancing, best-validation-loss checkpoint
selection, save/reload and Stage A adapter continuation on the RTX 5080. Each ran
two optimizer steps; these are functional proofs, not quality candidates. Adapter
continuation starts a fresh optimizer/scheduler. It does not restore an interrupted
Trainer checkpoint. Selection by loss is followed by separate output/meaning checks.

The existing 50-step English smoke adapter passed annotated checks on 23/25
English and 17/25 German challenge records, with EOS on all 50. Canonical term
correction passed 2/2 English and 0/2 German opportunities; no unsupported hint
was inserted in the 46 annotated absent-term opportunities. These small synthetic
counts do not establish performance on real German dictations.

A single continuous synthetic inventory report with exactly 2,048 non-padding
tokens passed forward/backward/optimizer steps at micro-batch 1 (7.3 GiB peak
allocated VRAM). Micro-batch 2 passed but reserved about 15.1 GiB and triggered
an allocator retry; micro-batch 4 failed with CUDA OOM. Use micro-batch 1 and
gradient accumulation 16 for a safe effective batch of 16 on full-length Stage A
examples. The original batch-4 smoke remains valid for its short examples.
The report is a physical memory probe, not corpus data or a quality evaluation.

## License

The training code is Apache-2.0. Datasets, base models, trained weights, and
other external artifacts retain their own licenses and must be registered in
`DATA_SOURCES.md` and generated manifests.

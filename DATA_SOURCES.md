# Third-party data registry

A source must have a reviewed
revision and license entry here before ingestion. Unknown terms are quarantined;
they are never inferred from repository names or dataset descriptions.

| Source | Revision | License | Allowed use | Redistribution |
|---|---|---|---|---|
| Sotto | `183cc8fd58532f13fa192980185214de1bcd5acc` | MIT declared in Hugging Face dataset metadata | Local model training and evaluation | Do not redistribute here; retain upstream attribution |
| Aawaaz | `5a8e62a11cebea4832057f304f12f8401d387b43` | MIT declared in Hugging Face dataset metadata | Local model training and evaluation | Do not redistribute here; retain upstream attribution |
| Deterministic OceanScribe targeted generator | `targeted-v1`; exact generator SHA-256 in each manifest | Apache-2.0; original project-authored templates, no third-party text | Local bilingual training diagnostics and synthetic challenge evaluation | No dataset redistribution here |
| Deterministic OceanScribe fidelity generator | `fidelity-v2`; exact generator SHA-256 and seed in each manifest/provenance file | Apache-2.0; original project-authored templates, no corpus or teacher input | Local number, self-correction, unit and formatting training diagnostics; separate challenge families | No dataset redistribution here |
| Private OceanScribe dictations | Local source revisions only | Private | Evaluation or training only with explicit owner approval | No |
| User-provided reviewed ChatGPT Pro translations and reference fixes | Frozen CSV SHA-256 and original source family in the pilot manifest; exact generating model/seed unavailable | `MIT AND LicenseRef-UserProvided-LocalTraining` | User explicitly supplied/generated these derivatives and requested local specialized transcript-cleanup training | No |

`LicenseRef-UserProvided-LocalTraining` records the user's authorization for this
local experiment, not a public MIT grant for generated translations. Upstream
MIT attribution remains attached. The original reviewed CSV is distinguished
from Luna's growing output; this pilot uses English KEEP/FIX and German PASS
from that original CSV. Other missing/REVIEW or unregistered records stay excluded.
OpenAI's [Europe Terms of Use](https://openai.com/policies/eu-terms-of-use/)
(updated January 16, 2026; reviewed October 1, 2026) assign output rights between
the user and OpenAI subject to applicable law and their usage restrictions.
This registry entry covers the requested local specialized cleanup experiment;
it does not assert permission for competing general-purpose models or publication.
Stochastic generated records retain a null seed and an explicit unavailable-seed
flag; deterministic augmentation seeds do not stand in for teacher sampling seeds.

Generated dataset manifests additionally record record counts, source revisions,
prompt-contract version, language/feature/length distributions, deduplication,
oversize exclusions, and a content SHA-256.

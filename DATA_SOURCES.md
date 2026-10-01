# Third-party data registry

A source must have a reviewed
revision and license entry here before ingestion. Unknown terms are quarantined;
they are never inferred from repository names or dataset descriptions.

| Source | Revision | License | Allowed use | Redistribution |
|---|---|---|---|---|
| Sotto | `183cc8fd58532f13fa192980185214de1bcd5acc` | MIT declared in Hugging Face dataset metadata | Local model training and evaluation | Do not redistribute here; retain upstream attribution |
| Aawaaz | `5a8e62a11cebea4832057f304f12f8401d387b43` | MIT declared in Hugging Face dataset metadata | Local model training and evaluation | Do not redistribute here; retain upstream attribution |
| Deterministic OceanScribe targeted generator | `targeted-v1`; exact generator SHA-256 in each manifest | Apache-2.0; original project-authored templates, no third-party text | Local bilingual training diagnostics and synthetic challenge evaluation | No dataset redistribution here |
| Private OceanScribe dictations | Local source revisions only | Private | Evaluation or training only with explicit owner approval | No |

Generated dataset manifests additionally record record counts, source revisions,
prompt-contract version, language/feature/length distributions, deduplication,
oversize exclusions, and a content SHA-256.

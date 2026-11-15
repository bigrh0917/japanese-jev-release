# Licenses and data provenance

This repository contains material under different licenses.

| What | License | Notes |
| --- | --- | --- |
| Code (`src/`, `scripts/`, `tests/`, `configs/`) | MIT ([LICENSE](LICENSE)) | |
| Self-built evaluation sets (`evalsets/`) | CC BY 4.0 | Written by Claude (Anthropic's AI). Please do not train on them |
| Synthetic data (`datasets/synthetic/`) | CC BY 4.0 | Written by Claude. Texts close to the evaluation sets were removed by character 3-gram overlap |
| LoRA weights (distributed on Hugging Face) | CC BY-SA 4.0 | The training data includes CC BY-SA datasets, so the same terms apply |
| Base models Qwen3-1.7B-Base, Qwen3-0.6B-Base | Apache-2.0 | Weights are not included; get them from Hugging Face |

## Public datasets

Neither the datasets nor their converted files are included in this repository. `japanese-jev build` fetches them from their distributors and converts them. Check each license before use. The table reflects what was checked when this project was built; the distributors are authoritative for current terms.

| Dataset | Use | License (when checked) |
| --- | --- | --- |
| JNLI (JGLUE) | training | CC BY-SA 4.0 |
| JSICK | training | CC BY-SA 4.0 |
| JSTS (JGLUE) | training | CC BY-SA 4.0 |
| KUCI | training | see the distributor |
| PAWS-X (Japanese) | training | see the distributor (attribution required) |
| MewsC-16-ja | training | see the distributor |
| SIB-200 | temperature calibration only | see the distributor |
| JCommonsenseQA (JGLUE) | evaluation only | CC BY-SA 4.0 |
| MASSIVE | evaluation only | see the distributor |
| JCoLA | evaluation only (one optional config uses its training split) | CC BY-SA 4.0 |
| livedoor news corpus | evaluation only | CC BY-ND. Derivatives cannot be redistributed, so converted files are not included |
| WRIME | evaluation only | research use only, no redistribution. Converted files are not included |

MARC-ja is not used because the underlying Amazon review corpus is no longer distributed.

## Comparison with Jev

The Jev numbers were obtained by sending part of the evaluation sets to the TypeSafe API (through OpenRouter). The individual answers and probabilities returned by the API are not included in this repository, and neither are the Jev outputs collected for the distillation experiment.

## Synthetic data and the use of AI

The synthetic data and the self-built evaluation sets were written by Claude. Anthropic explains that it allows Claude's outputs to be used to train models that do not compete with Anthropic's models (classifiers, for example). This model is a classification model that does not generate text.

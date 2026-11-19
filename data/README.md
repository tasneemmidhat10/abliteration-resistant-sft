---
language: [en]
size_categories: [1K<n<10K]
task_categories: [text-generation]
configs:
  - config_name: refusal
    data_files: wildguard.jsonl
  - config_name: coding
    data_files: tulu_coding.jsonl
  - config_name: instruction-following
    data_files: tulu_instruction-following.jsonl
---

# SFT data: refusal, coding, instruction following (3,000 samples)

Single-turn chat samples for supervised fine-tuning, 1,000 per domain. They are built by
[`src/arft/data.py`](../src/arft/data.py) with seed 0 and are fully reproducible.

```bash
source scripts/activate.sh
python -m arft.data            # writes the three files below into data/
```

You need a Hugging Face login that has accepted the terms of the gated dataset
[`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix).

## Files

| File | Rows | Domain | `task_type` | `is_refusal_example` | `source` |
|---|---|---|---|---|---|
| `wildguard.jsonl` | 1,000 | refusal | `safety` | `true` | `wildguardmix` |
| `tulu_coding.jsonl` | 1,000 | code | `coding` | `false` | `ai2-adapt-dev/personahub_code_v2_34999` |
| `tulu_instruction-following.jsonl` | 1,000 | general instruction following | `instruction-following` | `false` | `ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980` |
| **total** | **3,000** | | | | |

## Schema

Each line is one JSON object:

```json
{
  "id": "personahub_jk6upe9i68rsfa2635kvhg1f",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "task_type": "coding",
  "is_refusal_example": false,
  "source": "ai2-adapt-dev/personahub_code_v2_34999"
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique across all three files. Tülu rows keep the source record's own `id`. WildGuardMix has no id column, so its rows use the SHA-256 hex digest of the prompt. |
| `messages` | list | Exactly one `user` turn followed by one `assistant` turn. No system prompt. |
| `task_type` | string | The subset: `safety`, `coding` or `instruction-following`. |
| `is_refusal_example` | bool | `true` only for the WildGuardMix refusals. |
| `source` | string | The upstream dataset (WildGuardMix) or the Tülu `source` value. |

## Sources and selection

| Subset | Upstream dataset | Split / config | Filter | Eligible pool | Selected |
|---|---|---|---|---|---|
| refusal | [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) | `wildguardtrain`, `train` (86,759 rows) | `adversarial == False`, `prompt_harm_label == "harmful"`, `response_refusal_label == "refusal"`, `response` not null | 6,062 | 1,000 |
| coding | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `train` | `source == "ai2-adapt-dev/personahub_code_v2_34999"`, single-turn (`user` → `assistant`) | 34,999 | 1,000 |
| instruction-following | [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `train` | `source == "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"`, single-turn (`user` → `assistant`) | 29,980 | 1,000 |

Selection: each filtered pool is shuffled with `datasets.Dataset.shuffle(seed=0)`, and the first 1,000 rows
are kept. No deduplication, decontamination or junk filtering is applied.

The two Tülu subsets are the Persona code and Persona instruction-following components of the Tülu 3 SFT mixture.
AI2 also publishes these components as standalone datasets,
[`allenai/tulu-3-sft-personas-code`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code) and
[`allenai/tulu-3-sft-personas-instruction-following`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following).
This build loads them through the mixture. Row-for-row identity with the standalone releases has not been checked.

## Statistics

Character lengths, median / max:

| Subset | Prompt | Response |
|---|---|---|
| refusal | 141 / 847 | 628 / 6,222 |
| coding | 924 / 2,533 | 335 / 2,873 |
| instruction-following | 276 / 1,126 | 1,300 / 29,132 |

WildGuardMix harm subcategories among the 1,000 refusals:

| Subcategory | n |
|---|---|
| social_stereotypes_and_unfair_discrimination | 129 |
| violence_and_physical_harm | 115 |
| disseminating_false_or_misleading_information_encouraging_disinformation_campaigns | 102 |
| sexual_content | 86 |
| sensitive_information_organization_government | 85 |
| toxic_language_hate_speech | 84 |
| defamation_encouraging_unethical_or_unsafe_actions | 78 |
| mental_health_over-reliance_crisis | 73 |
| copyright_violations | 70 |
| cyberattack | 64 |
| fraud_assisting_illegal_activities | 47 |
| causing_material_harm_by_disseminating_misinformation | 34 |
| private_information_individual | 33 |

## Known limitations

- **Refusals are non-adversarial only.** Jailbreak-style (adversarial) WildGuardMix prompts are excluded, so the
  refusal subset covers only plainly stated harmful requests.
- **Harmful content.** The refusal prompts are harmful requests (violence, hate speech, sexual content and so on).
  The responses are refusals, but the prompts themselves are offensive.
- **No decontamination.** Prompts are not checked against evaluation benchmarks.
- **Synthetic responses.** All three subsets contain model-generated responses, which have not been checked by a
  person.

## Licence and citation

WildGuardMix and the Tülu 3 SFT mixture are both released under ODC-BY-1.0. Their responses were generated by
third-party models, so those models' terms may also apply. Check the upstream dataset cards before redistributing
the data. WildGuardMix is gated: anyone using this data should accept its terms on Hugging Face.

- Han et al., 2024. *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.*
  arXiv:2406.18495.
- Lambert et al., 2024. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv:2411.15124.

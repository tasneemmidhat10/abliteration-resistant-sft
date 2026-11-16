# Data card — POC training data (Step 1)

Built by [`src/arft/data.py`](../src/arft/data.py) with seed 0. To rebuild:

```bash
source scripts/activate.sh
sbatch scripts/slurm/build_data.sh      # or: python -m arft.data
```

You need a Hugging Face login with the terms accepted for the gated datasets
[`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) and
[`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench).

## What is in git and what is not

| File | In git | Content |
|---|---|---|
| `data/train.jsonl`, `data/val.jsonl` | no (~30 MB, rebuilt by the script) | the full splits |
| `data/stats.json` | yes | counts at each filtering stage, per split/source, length statistics |
| `data/samples.jsonl` | yes | 20 train examples each of coding and math, for reading |
| `data/samples_wildguardmix.jsonl` | no (gated dataset, public repo) | 20 train examples each of refusal and general |
| `data/eval_indices.json` | yes | frozen random evaluation subsets (see [Evaluation](#evaluation-subsets)) |

## Composition

| Stratum | Source | Selection | Train | Val | `task_type` | `is_refusal_example` |
|---|---|---|---|---|---|---|
| refusal | WildGuardMix `wildguardtrain` | non-adversarial, harmful prompt, refusal response | 1,387 | 154 | `safety` | `true` |
| general | WildGuardMix `wildguardtrain` | non-adversarial, unharmful prompt, compliant response | 1,387 | 154 | `general` | `false` |
| coding | Tülu 3 SFT mixture, `ai2-adapt-dev/personahub_code_v2_34999` | single-turn | 3,600 | 400 | `coding` | `false` |
| math | Tülu 3 SFT mixture, `ai2-adapt-dev/personahub_math_v5_regen_149960` | single-turn | 3,600 | 400 | `math` | `false` |
| **total** | | | **9,974** | **1,108** | | |

Every row follows the handbook's uniform schema (§6.5). The labels are never shown to the model:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "task_type": "coding", "is_refusal_example": false, "source": "ai2-adapt-dev/personahub_code_v2_34999"}
```

## Pipeline and how it differs from the handbook

1. **Exact Tülu source names.** The handbook's substring `personahub_math` also matches
   `tulu_v3.9_personahub_math_interm_algebra_20k`. That is the Algebra subset, which is reserved for the
   paper phase, and it would have been mixed in without any error. The script matches exact names and stops
   with an error if a name disappears upstream. Tülu's own WildGuardMix component is excluded because
   only the two named sources are used.
2. **Exact-duplicate removal.** Prompts are compared after lower-casing and collapsing whitespace, within and
   across sources. This removed 367 general and 8 coding rows.
3. **Decontamination.** A training prompt is dropped if it shares any 13-word n-gram, or its full normalised
   text, with an evaluation or attacker prompt: GSM8K test, HumanEval, WildGuardMix-test, AdvBench. This
   removed 9 + 1 refusal, 2 general and 3 coding rows. Alpaca (the attacker's harmless set) is not checked.
4. **Length matching of refusal vs general.** In the clean pool, harmful prompts are about twice as long as
   harmless ones (median 22 vs 11 words). A word-count threshold alone separated them with **83.9 %**
   accuracy, so the policy head could have learned length instead of harm. Both strata are now sampled
   with equal counts per word-length bin. The threshold accuracy drops to **50.7 %** (chance), and the
   medians are 13 and 13 words. The cost is 1,541 rows per stratum instead of 2,000: harmless prompts
   longer than 16 words are rare. Use `--no-length-match` to get the handbook's 2k/2k version.
5. **Stratified split.** 10 % of each stratum goes to val, so the val class balance equals the train class balance.

All counts above are recorded in `data/stats.json` under `filtering`.

## Known limitations

- **Style gap between Persona and WildGuardMix.** Coding and math prompts are long, synthetic and
  persona-framed (median 147 and 203 words). The WildGuardMix prompts are short (13 words). Length matching
  cannot close this gap. For the policy head, the only clean contrast is **refusal vs general**, which come
  from the same source and are length-matched. Check head accuracy on that pair separately, not only overall.
- **No example carries both labels.** Refusal rows have no capability label (`cap_label = -100`), and coding
  and math rows are never refusals. The two heads do share the general, coding and math rows: the policy
  head sees them as negatives. However, the data never pairs "refuse" with a capability class. Nothing here
  forces the two heads onto shared directions of `z`, so they may read from orthogonal subspaces (handbook,
  Step 3 caveat). **Measuring the head geometry after training is therefore a required part of Step 3, not an
  extra:** the cosine between `W_p[1] − W_p[0]` and the attacker's refusal direction `r`, and the principal
  angles between the policy and capability head weights.
- **Selection effect of length matching.** The refusal rows kept are the shorter harmful prompts, which tend to
  be blunter requests ("Generate a plan for kidnapping …").
- **Label noise.** WildGuardMix labels are model- and annotator-generated. Some "compliance" responses read as
  partial refusals, e.g. a chess-strategy answer that lectures about manipulation.
- **Contamination check is lexical.** Paraphrased test items are not caught.

## Evaluation subsets

`data/eval_indices.json` freezes seeded random subsets of the test splits:

- `gsm8k`: 250 row indices into `openai/gsm8k` (main, test). Pass them as
  `lm_eval.simple_evaluate(samples={"gsm8k": [...]})`. Do not use `limit=250`, which takes the
  *first* 250 items (verified in lm-eval 0.4.13).
- `wildguardtest_harmful`: 200 indices of non-adversarial harmful prompts in WildGuardMix-test (413 available),
  for the refusal evaluation.
- HumanEval: all 164 problems.

**What these sizes can show.** At n = 164–250, the standard error of one accuracy is about 3 points. The
difference between two models scored independently has a 95 % interval of about **±7 to ±11 points**, depending
on accuracy. Every model is evaluated on the *same* items, so compare models **paired by item** (paired
bootstrap or McNemar's test on the per-item outcomes from `log_samples=True`), which gives a tighter
interval. With one seed, differences below about 10 points between coupled and baseline should be reported as
inconclusive.

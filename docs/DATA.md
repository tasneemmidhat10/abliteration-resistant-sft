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
| refusal | WildGuardMix `wildguardtrain` | non-adversarial, harmful prompt, refusal response | 1,800 | 200 | `safety` | `true` |
| general | WildGuardMix `wildguardtrain` | non-adversarial, unharmful prompt, compliant response | 1,800 | 200 | `general` | `false` |
| coding | Tülu 3 SFT mixture, `ai2-adapt-dev/personahub_code_v2_34999` | single-turn | 3,600 | 400 | `coding` | `false` |
| math | Tülu 3 SFT mixture, `ai2-adapt-dev/personahub_math_v5_regen_149960` | single-turn | 3,600 | 400 | `math` | `false` |
| **total** | | | **10,800** | **1,200** | | |

These are the handbook's POC sizes (2k / 2k / 4k / 4k, 90/10). Every row follows the handbook's uniform schema
(§6.5). The labels are never shown to the model:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "task_type": "coding", "is_refusal_example": false, "source": "ai2-adapt-dev/personahub_code_v2_34999"}
```

## Pipeline

Each stratum goes through the same steps, in this order. All counts are in `data/stats.json` under `filtering`.

| Step | Refusal | General | Coding | Math |
|---|---|---|---|---|
| Candidates | 6,062 | 3,898 | 34,999 | 149,960 |
| Junk prompts dropped | 0 | 0 | 11 | 0 |
| Whole-row duplicates dropped | 0 | 0 | 0 | 0 |
| Contaminated prompts dropped | 10 | 2 | 8 | 2 |
| Clean pool | 6,052 | 3,896 | 34,980 | 149,958 |
| Selected (random draw) | 2,000 | 2,000 | 4,000 | 4,000 |

1. **Exact Tülu source names.** The handbook's substring `personahub_math` also matches
   `tulu_v3.9_personahub_math_interm_algebra_20k`. That is the Algebra subset, which is reserved for the
   paper phase, and it would have been mixed in without any error. The script matches exact names and stops
   with an error if a name disappears upstream. Tülu's own WildGuardMix component is excluded because only the
   two named sources are used.
2. **Junk prompts.** A row is dropped if its *whole* prompt is a refusal sentence ("I'm sorry, I can('t|not)
   assist with that( request)."). Tülu Persona code contains 11 such rows, answered with stub functions. The
   match is on the whole prompt, so ordinary prompts that merely start with similar words are kept.
3. **Whole-row duplicates.** A row is dropped only if both its prompt and its response repeat an earlier row
   (after lower-casing and collapsing whitespace), within or across sources. This currently drops nothing; the
   only whole-row duplicate was inside the junk group. Rows that share a prompt but have different responses are
   **kept**: WildGuardMix often stores several responses per prompt. See [Known limitations](#known-limitations).
4. **Decontamination.** A training prompt is dropped if its normalised text equals, or shares any 13-word
   n-gram with, an evaluation prompt (GSM8K test, HumanEval, WildGuardMix-test) or an attacker prompt (AdvBench,
   Alpaca no-input instructions, the whole set). Most Alpaca instructions are shorter than 13 words, so for
   them only the exact match applies. The Alpaca hits in coding and math are template phrases, not copied
   prompts.
5. **Selection.** Each clean pool is shuffled with its own seeded RNG and cut to its target, so the selection is
   a uniform random draw, and changing one pool never reshuffles another.
6. **Stratified, grouped split.** 10 % of every stratum goes to val (exactly 200 / 200 / 400 / 400), so the val
   class balance equals the train class balance. Rows that share a normalised prompt always go to the same
   split: no val prompt appears in train (`split_prompt_overlap: 0` in `stats.json`; the build asserts it).

## Provenance and references

This section separates what the handbook asks for from what this repository chose. All citations were checked
against the papers on 2026-11-18.

| Step | Where it comes from | References |
|---|---|---|
| Decontamination | **Handbook**, Step 1 "Paper extras" ([HANDBOOK.md](HANDBOOK.md), line 527): "Contamination check: n-gram overlap of training data with HumanEval / GSM8K test items". §6.4 lists the attacker data (AdvBench, Alpaca) as "never used in training". Extending the check to WildGuardMix-test, AdvBench and Alpaca, and running it in the POC, are repository choices. | [1] [2] |
| 13-word n-grams on lower-cased words, punctuation ignored | Repository choice that follows GPT-3 [1]. Tülu 3, the source of our coding and math rows, decontaminated its own mixture with 8-gram matching [2]. Llama 2 used a different method: token sequences of 10 or more tokens with skip-grams, not 13-grams [3]. | [1] [2] [3] |
| Whole-row deduplication | Not in the handbook; repository choice. Prompt-level deduplication was tried and is switched off for now (see limitations). | [4] |
| Junk-prompt filter | Not in the handbook; found by reading the data (Step 1 audit). | — |
| Length-shortcut diagnostic | Repository choice. An earlier build length-matched refusal and general prompts (commit 710d44f). That step had no source in the handbook or the literature, and it was **removed on 2026-11-18**. The measurement stays as a diagnostic. | motivation: [5] [6] |
| Stratified split | The handbook asks for a "90/10 train/val" split. Stratifying per stratum is a repository choice. | [7] |
| Grouping by prompt in the split | Repository choice, needed once same-prompt rows are kept. Same idea as scikit-learn's `StratifiedGroupKFold` (stratified folds with non-overlapping groups). | [8] |

**What the references say, in brief**

- [1] GPT-3 counted a "gram" as a lower-cased, whitespace-delimited word without punctuation and used 13-grams.
  It tried to remove overlaps from its training data (a bug left some in) and split each benchmark into clean
  and dirty subsets (§4, Appendix C).
- [2] Tülu 3 checked prompts only, with 8-gram matching. A training instance counted as contaminated when more
  than 50 % of a test instance's tokens matched it, and contaminated instances were removed from the training
  mix (§3.2; HumanEval appears in Table 8).
- [4] Deduplication reduces overlap between training and evaluation data; the authors report that such overlap
  affects over 4 % of the validation sets of standard datasets.
- [5] [6] Models learn whatever simple cue separates the labels. In SNLI, hypothesis *length* alone predicts the
  label [5]; [6] describes such "shortcuts" in general. This is why the length-only accuracy is reported.
- [7] Recommends stratified 10-fold cross-validation for model selection on real-world datasets.

1. T. B. Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020. arXiv:2005.14165. §4 and Appendix C.
2. N. Lambert et al. *Tülu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv:2411.15124 (2024). §3.2 "Prompt Decontamination".
3. H. Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288 (2023). Appendix A.6.
4. K. Lee et al. *Deduplicating Training Data Makes Language Models Better.* ACL 2022, pp. 8424–8445. doi:10.18653/v1/2022.acl-long.577.
5. S. Gururangan et al. *Annotation Artifacts in Natural Language Inference Data.* NAACL 2018. arXiv:1803.02324.
6. R. Geirhos et al. *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2(11):665–673 (2020). doi:10.1038/s42256-020-00257-z.
7. R. Kohavi. *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI 1995, pp. 1137–1144.
8. scikit-learn, `sklearn.model_selection.StratifiedGroupKFold`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Known limitations

- **Prompt length predicts refusal vs general.** Harmful prompts are about twice as long as harmless ones
  (median 22 vs 11 words). A single word-count threshold separates the selected refusal and general rows with
  **85.2 %** accuracy (`length_only_acc_refusal_vs_general` in `stats.json`; 50 % is chance), so the policy head
  can partly learn length instead of harm. Report the head's accuracy on a length-controlled subset of the
  evaluation as well, not only overall.
- **Repeated prompts are kept.** In the selected general rows, 90 prompts occur 2–4 times (191 rows), each time
  with a different response. These prompts get extra weight in training. The split keeps each of them on one
  side, so validation is not affected.
- **Lecture-style "compliance" rows.** About 8.4 % of the selected general rows (168 of 2,000, by a heuristic on
  the opening words) open like a refusal or a safety lecture ("I must emphasize…") although they are labelled as
  compliance with a harmless prompt. These rows teach over-refusal on harmless prompts. They are not filtered.
  The share is higher than in earlier builds because every response to a repeated prompt is now kept, and
  repeated prompts carry such responses about 3.5 times as often as other prompts.
- **Style gap between Persona and WildGuardMix.** Coding and math prompts are long, synthetic and
  persona-framed (median 148 and 205 words). The WildGuardMix prompts are short (11–22 words). For the policy
  head, the only clean contrast is **refusal vs general**, which come from the same source. Check head accuracy
  on that pair separately, not only overall.
- **No example carries both labels.** Refusal rows have no capability label (`cap_label = -100`), and coding
  and math rows are never refusals. The two heads do share the general, coding and math rows: the policy head
  sees them as negatives. However, the data never pairs "refuse" with a capability class. Nothing here forces the
  two heads onto shared directions of `z`, so they may read from orthogonal subspaces (handbook, Step 3 caveat).
  **Measuring the head geometry after training is therefore a required part of Step 3, not an extra:** the
  cosine between `W_p[1] − W_p[0]` and the attacker's refusal direction `r`, and the principal angles between the
  policy and capability head weights.
- **Label noise.** WildGuardMix labels are model- and annotator-generated.
- **Contamination check is lexical.** Paraphrased test items, and paraphrases of short attacker prompts, are not
  caught.

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

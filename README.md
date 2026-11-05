# GEPA prompt optimization for Hebrew question answering

Does automatic prompt optimization work in Hebrew? Most published results for
[GEPA](https://arxiv.org/abs/2507.19457) (reflective prompt evolution) come from English
tasks. This project tests whether they transfer to a morphologically rich language, on
Hebrew question answering over documents, and what the gains cost.

The pipeline is built on [DSPy](https://dspy.ai) (`dspy.GEPA`). All model calls go through
[OpenRouter](https://openrouter.ai), and every run logs its models, seed, budget, prompts,
scores, token counts and dollar cost.

## Research questions

| # | Question |
|---|---|
| RQ1 | Does GEPA beat a hand-written Hebrew baseline prompt? |
| RQ2 | Does Hebrew diagnostic feedback help GEPA more than a bare score? |
| RQ3 | Does a Hebrew seed prompt do better than an English one? |
| RQ4 | Is the gain larger for a mid-size task model (Qwen 3.6, ~35B) than for a strong model? |
| RQ5 | What does each point of improvement cost (rollouts, tokens, dollars)? |

## Models

| Role | Model | Notes |
|---|---|---|
| Task model (the prompt being optimized) | `qwen/qwen3.6-35b-a3b` | temperature 0; 16K max tokens on the abstractive task, because Qwen thinks at length on long documents |
| Reflection model (proposes new prompts) | `anthropic/claude-sonnet-5.5` | temperature 1.0 |
| Judge (scores abstractive answers) | `anthropic/claude-sonnet-5.5` | frozen after validation (stage 2) |
| Second judge, evaluated alongside | `google/gemma-4-31b-it` | |

The judges come from outside the Qwen family, so the task model is never graded by its own family.

## Project stages

1. **Pilot on HeQ (extractive QA).** Proves the pipeline end to end on a task with an automatic metric. Done.
2. **The judge.** Builds and validates an LLM judge for abstractive answers against human ratings. Done; validation on fresh Qwen answers is waiting for human ratings.
3. **GEPA on abstractive QA,** with the judge as the metric. First runs done.

---

## Stage 1: pilot on HeQ

[HeQ](https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset) (v1.1) is a Hebrew
extractive QA dataset: the answer is a span of the passage. The pilot draws 100 train,
100 validation and 300 test questions (answerable only, seed 42), each set from the matching
official split.

- **Metric:** TLNLS (token-level normalized Levenshtein similarity, from the HeQ paper),
  plus exact match (EM) and F1. Answers are normalized for Hebrew prefixes and punctuation
  before scoring.
- **Feedback for GEPA:** a Hebrew diagnosis of each error: an empty answer, a difference
  only in a prefix letter (ו, ב, כ, ל, מ, ש, ה), an answer too long or too short, text not
  copied from the passage, or the wrong span.
- **Budget:** GEPA `auto="light"`, about 860 metric calls per run.

### Results (300 test questions)

| Seed prompt | Prompt | EM | F1 | TLNLS | Optimization cost |
|---|---|---|---|---|---|
| Hebrew | hand-written seed | 84.33 | 94.02 | 94.72 | |
| Hebrew | GEPA | **86.33** | **94.66** | **95.72** | $1.48 |
| English | hand-written seed | 84.67 | 93.65 | 94.72 | |
| English | GEPA | **87.67** | **95.31** | **95.82** | $1.20 |

GEPA improved both seeds by about 1 TLNLS point and 2–3 EM points, for under $1.50 per run.
The pilot was near its ceiling (the seed already scores ~95 TLNLS), so it served mainly to
prove the pipeline, logging and cost tracking, not to answer the research questions.

---

## Stage 2: abstractive QA and the judge

### The dataset

132 Hebrew questions on 121 documents (journalistic articles, Knesset committee protocols,
encyclopedic entries). Each record holds one model-written answer, broken into claims, each
claim backed by verbatim quotes from the document. Four human annotators rated each answer
0–2 on five criteria (the dataset's "Part II" guidelines):

| Criterion | What it checks |
|---|---|
| Coherence | natural, clear, well-structured Hebrew |
| Relevance | answers exactly what was asked, without padding |
| Completeness | all the information the question needs |
| Faithfulness | nothing beyond the document: no inventions, no outside knowledge |
| Attribution | every claim backed by a quote that actually supports it |

Questions are tagged with a level ("Part I" guidelines):

| Level | Meaning | Questions |
|---|---|---|
| 0 | the document does not contain the answer; the right answer is a refusal | 39 |
| 1 | a single explicit detail | 45 |
| 2 | 2–3 details from different places in the document, combined | 48 |

Level 3 was removed from the dataset by design.

**Split:** by document, so all questions on a document land in the same set: 29 train,
30 validation, 72 test (seed 42), with unanswerable questions spread evenly. Train and
validation together are the development set; all judge tuning used only those, and the
test set was kept for reporting.

**The task output** matches the dataset: an answer plus a list of claims, each with
verbatim quotes. A refusal is the fixed sentence "המידע המבוקש אינו מופיע במסמך." with no
claims.

### The judge

The judge ([hebqa/judge](hebqa/judge)) follows the annotators' guidelines nearly verbatim,
in Hebrew. It does not see a reference answer, just as the annotators didn't. It receives:

- the document, the question and its level;
- the answer, with each claim and its quotes;
- for each quote, a mechanical check of whether it appears in the document word for word
  (tolerant of whitespace, quote-mark style and ellipses).

For each criterion it writes a Hebrew rationale, then a 0–2 score. The mean of the five
scores, mapped to 0–1, is the GEPA metric. The rationales are the feedback GEPA's
reflection model reads.

**Calibration.** The rubric adds clarifications where the first judge runs disagreed with
the annotators. For example, paraphrase and direct inference count as faithful; a quote
longer than needed is not "too general"; and a score below 2 must point to a problem worth
correcting. Eight disputed cases were sent to an annotator. Four of the answers became
worked examples in the judge's instructions. All eight come from the development set,
so the test set stays clean.

**Claim-by-claim check (tried, off by default).** A separate call checking each claim
against its quotes made Sonnet stricter on attribution and lowered agreement with the
annotators. It is kept as an option (`Judge(lm, check_claims=True)`).

### Validation against the annotators (72 test questions)

Agreement is Krippendorff's α (interval): the judge against each annotator, compared with
the annotators against each other.

| Criterion | Annotator vs annotator | Sonnet (frozen judge) | Gemma 4 |
|---|---|---|---|
| Coherence | 0.01 | 0.16 | 0.22 |
| Relevance | 0.32 | 0.40 | −0.01 |
| Completeness | 0.64 | 0.80 | 0.80 |
| Faithfulness | 0.55 | 0.56 | 0.73 |
| Attribution | 0.46 | 0.31 | 0.34 |

Sonnet agrees with the annotators at least as well as they agree with each other on four of
five criteria. **Gemma almost never scores below 2:** it caught 1 of 22 relevance problems
and none of 6 coherence problems. Its high faithfulness agreement comes from the dataset's
answers being mostly correct.

Attribution is the weakest criterion, and part of the reason is in the data (below).

### Data issues found

- **Misaligned quotes.** The dataset stores each quote as a sentence number plus text
  resolved automatically. In some records the text comes from the wrong sentence, so the
  stored quote is unrelated to its claim, although the annotators rated the claim as
  supported. In record `b1ea313e` the correct sentences are 31 and 38 but 23 and 26 are
  stored. The shift is not constant, so it can't be repaired in code. That record is fixed
  by hand with quotes an annotator located (`QUOTE_FIXES` in
  [hebqa/data/abstractive.py](hebqa/data/abstractive.py)). The judge flags six more
  suspects (`b015a9bd`, `fdc3bf1f`, `2bceead9`, `d80d3f40`, `c43144b6`, `96e6c6fc`), which
  await human review. They are not excluded on the judge's word alone, since that would
  inflate its agreement by construction.
- **Garbled document.** Record `673d9846` is excluded (`EXCLUDED`). It is dropped after the
  split, so the other records keep their sets.

### Human ratings of Qwen answers (pending)

The dataset's answers come from a strong model and are almost all 2s, while GEPA will see
Qwen's answers, which have more real problems.
[scripts/build_rating_pool.py](scripts/build_rating_pool.py) draws 39 development questions
(stratified by level) and answers them with the Qwen seed prompt.
[labeling/judge_labeling.html](labeling/judge_labeling.html) is the rating page: the five
criteria, a required correction when a score is below 2, and quotes highlighted in the
document. Once these are rated, both judges will be validated on them.

---

## Stage 3: GEPA with the judge as the metric

Two GEPA runs, identical except for the judge: Hebrew seed prompt, Qwen task model, Sonnet
reflection, 29 train and 30 validation questions, `auto="light"` (about 500 metric calls).
The seed prompt and both optimized prompts were then scored on the 72 test questions by
**both** judges, so each prompt is also graded by the judge it was not optimized against.

### Optimization

| | Judged by Sonnet | Judged by Gemma 4 |
|---|---|---|
| Validation score, seed → best | 0.903 → **0.960** | 0.960 → 0.960 |
| Candidates found | 13 | 1 (the seed) |
| Cost | $11.30 (judge $8.86, task $1.54, reflection $0.90) | $0.09 |

The Gemma run never optimized anything. Gemma gave a perfect score to every training
minibatch, so GEPA skipped all 157 iterations without calling the reflection model.

### Test set (72 questions)

| Prompt | Scored by Sonnet | Scored by Gemma |
|---|---|---|
| Seed | 0.958 | 0.999 |
| GEPA, Sonnet judge | **0.979** | 0.994 |
| GEPA, Gemma judge (= seed) | 0.958 | 0.999 |

- Under Sonnet, the optimized prompt improved 17 answers and worsened 4. The paired
  bootstrap 95% interval for the gain is **[−0.008, +0.047]**, so with 72 questions the gain
  is not statistically clear.
- The gain is mostly in relevance (1.88 → 1.97) and coherence (1.92 → 1.96): answers
  became shorter (109 → 73 characters on average). It is concentrated in level 2
  questions (0.89 → 0.93).
- Gemma does not separate the prompts, so the gain has no independent confirmation yet.
- One new failure: the optimized prompt refused a question the document does answer.
  It traces to a rule GEPA wrote, "don't infer from indirect hints".
- The evolved prompt names specifics from training questions, a sign of fitting to the
  training set.

### Conclusions so far

1. GEPA works end to end on Hebrew abstractive QA with an LLM judge, at about $11 per light run.
2. **The judge must discriminate.** A lenient judge (Gemma 4) gives GEPA no signal, and the
   optimization never starts.
3. The gain measured by the optimizing judge needs human confirmation. The planned check is a
   blind side-by-side human comparison of the 21 test questions where the two prompts' answers differ.

## Next steps

- Human ratings of the 39 Qwen answers, then validate both judges on them.
- Human review of the six suspect misaligned-quote records.
- Blind human comparison of seed and optimized answers on test.
- Repeat runs with different seeds to measure run-to-run variance before reporting RQ1.
- Score-only feedback (RQ2), English seed (RQ3), a strong task model (RQ4).

---

## Repository layout

```
hebqa/
  config.py           paths, split sizes, model settings, GEPA settings
  lm.py               OpenRouter clients; token and cost accounting
  claims.py           Claim model, verbatim quote check, rendering for the judge
  optimize.py         GEPA wrapper and candidate summary
  evaluation.py       extractive evaluation (EM, F1, TLNLS)
  data/               HeQ loader, abstractive loader (with data fixes), document-level split
  metrics/            Hebrew normalization, TLNLS, Hebrew feedback, GEPA metric for HeQ
  programs/           DSPy signatures and the Hebrew/English seed prompts
  judge/              rubric (Hebrew), judge, agreement statistics, GEPA judge metric
scripts/
  eval_baseline.py         stage 1: seed prompt on HeQ
  run_gepa.py              stage 1: GEPA on HeQ
  validate_judge.py        stage 2: judges vs the annotators' ratings
  build_rating_pool.py     stage 2: Qwen answers for human rating
  run_gepa_abstractive.py  stage 3: GEPA with a judge as the metric
  compare_gepa_judges.py   stage 3: score prompts on test with every judge
labeling/judge_labeling.html   rating page for human annotators
data/                          HeQ v1.1 and the abstractive QA set
tests/                         pytest suite (no API calls)
```

Run outputs go to `runs/<timestamp>_<name>/` (not in the repository): `run.json` with the
configuration, scores and per-model usage, plus per-item JSONL files.

## Setup and usage

```bash
pip install -e ".[dev]"
cp .env.example .env    # then set OPENROUTER_API_KEY
python -m pytest        # offline tests
```

Run from the project root:

```bash
# Stage 1
python -m scripts.eval_baseline --seed-lang he
python -m scripts.run_gepa --seed-lang he --feedback hebrew --auto light

# Stage 2
python -m scripts.validate_judge --judges sonnet gemma
python -m scripts.build_rating_pool --n 40

# Stage 3
python -m scripts.run_gepa_abstractive --judge sonnet
python -m scripts.run_gepa_abstractive --judge gemma
python -m scripts.compare_gepa_judges runs/<sonnet run> runs/<gemma run>
```

Approximate costs: a judge validation run is about $3.30 (Sonnet) and $0.10 (Gemma); a light
GEPA run with the Sonnet judge is about $11; the test comparison is about $2.70.

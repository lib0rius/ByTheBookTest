---
pretty_name: ByTheBookTest
license: cc-by-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
task_ids:
  - natural-language-inference
tags:
  - knowledge-conflict
  - context-faithfulness
  - logical-reasoning
  - deduction
  - abstention
  - synthetic
  - z3
configs:
  - config_name: level1
    default: true
    data_files:
      - split: test
        path: level1/claims.jsonl
  - config_name: level2
    data_files:
      - split: test
        path: level2-noise10/claims.jsonl
  - config_name: level3
    data_files:
      - split: test
        path: level3-noise20/claims.jsonl
  - config_name: level1-rulebooks
    data_files:
      - split: test
        path: level1/kbs.jsonl
  - config_name: level2-rulebooks
    data_files:
      - split: test
        path: level2-noise10/kbs.jsonl
  - config_name: level3-rulebooks
    data_files:
      - split: test
        path: level3-noise20/kbs.jsonl
  - config_name: world-probe
    data_files:
      - split: test
        path: world-probe/claims.jsonl
---

# ByTheBookTest

**Does your model follow the document, or what it already knows?**

Each item is a *rulebook* (a list of English statements that defines the world of the task) and a *claim*.
Using only the rulebook, the model must answer:

| Answer | Meaning |
|---|---|
| `TRUE` | the claim follows from the rulebook |
| `FALSE` | the negation of the claim follows from the rulebook |
| `UNKNOWN` | neither follows: the rulebook is silent |

Some rulebooks deliberately contradict the real world. One of them states *"A cactus grows under the sea."*;
asked *"A cactus does not grow under the sea."*, the right answer is `FALSE`, whatever the model knows about
cacti. When the rulebook says nothing, as for *"A helium balloon rises when released."* in the same rulebook,
the right answer is `UNKNOWN`, even though the claim is true in the real world.

Every expected answer was computed by construction and then checked independently by the
[Z3](https://github.com/Z3Prover/z3) theorem prover on the complete rulebook.

- **Code and generator:** [github.com/lib0rius/ByTheBookTest](https://github.com/lib0rius/ByTheBookTest)
- **Article, for a general reader:** [article/article.md](https://github.com/lib0rius/ByTheBookTest/blob/main/article/article.md)
- **Author:** Claudio Martines (lib0rius)

## Contents

12 rulebooks and 2,600 claims in three levels, plus a separate knowledge probe with no rulebook. All of it
was generated with ByTheBookTest v0.1.0 and seed 42.

| Config | Rulebooks × claims | Statements per rulebook | Prompt length, median | Reasoning steps | Reasoning groups |
|---|---|---|---|---|---|
| `level1` | 4 × 75 | 121–124 | 1,621 tokens | 0–1 | 3 |
| `level2` | 4 × 225 | 751–764 | 9,062 tokens | 0–3 | 9 |
| `level3` | 4 × 350 | 2,374–2,488 | 26,620 tokens | 0–6 | 14 |
| `world-probe` | no rulebook, 1,580 sentences | — | — | — | — |

Prompt lengths are the input tokens of the tested model, d1-2B, for the whole request: rulebook,
instructions and claim.

- **Reasoning groups.** Level 1: facts, rule chains, "can"/"must". Level 2 adds negation, conditions, "or",
  "if and only if", exceptions and dates. Level 3 adds relations, "some", counts, "every" and shared chains.
- **Distractors.** Statements that no claim needs. Level 1 has none. Level 2 keeps 10% of its distractors,
  which are unrelated to the claims. Level 3 keeps 20%, chosen to look like useful statements. The rest were
  removed at random, and Z3 confirmed every answer again after the removal.
- **Expected answers.** 858 `TRUE`, 858 `FALSE` and 884 `UNKNOWN`.

### Every group of 100 claims has the same mix

Each of the 26 reasoning groups (3 + 9 + 14) has 100 claims, split by expected answer and by relation with
the real world:

| Expected answer | Conflict | Agree | Silent on a real fact | Invented words | Total |
|---|---:|---:|---:|---:|---:|
| `TRUE` | 18 | 5 | — | 10 | 33 |
| `FALSE` | 18 | 5 | — | 10 | 33 |
| `UNKNOWN` | — | — | 12 + 12 | 10 | 34 |

- **Conflict:** the rulebook gives an answer, and the real world says the opposite.
- **Agree:** the rulebook gives an answer, and the real world says the same. These claims are the reference
  for the conflict claims, and they block a shortcut: a model that always answers the opposite of what it
  knows is right on every conflict claim but wrong on every agreeing one.
- **Silent on a real fact:** the rulebook says nothing. The claim is true in the real world (12 claims) or
  false (12 claims), so these claims can only be answered from outside knowledge.
- **Invented words:** the same reasoning with invented names and properties (*"Gikrustok flies."*), so
  there is nothing to know.

Across the 26 groups this gives 26 × 33 = 858 `TRUE`, 858 `FALSE` and 26 × 34 = 884 `UNKNOWN`.

### Knowledge probe (`world-probe`)

The probe measures which facts a model knows, with no rulebook. It covers 790 statements: 764 from the
curated fact tables (the true version of each fact and its false versions) and 26 abilities (*"A penguin can
fly."*). Each statement comes as an affirmative and a negative sentence, 1,580 sentences in all. The
expected answer is the truth in the real world: 790 `TRUE` and 790 `FALSE`. A model "knows" a statement if
it answers both sentences correctly.

## Fields

### Claims (`level1`, `level2`, `level3`, `world-probe`)

| Field | Meaning |
|---|---|
| `claim_id` | identifier, for example `L1-KB1-C0006` |
| `kb_id` | the rulebook the claim belongs to (`W-KB1` for the probe, which has no rulebook) |
| `level` | 1, 2 or 3 (0 for the probe) |
| `group`, `subtype` | the kind of reasoning needed, and the variant it was built from |
| `text` | the claim in English, as shown to the model |
| `logic` | the claim as a formula, as checked by Z3 (for the probe: `pos` or `neg`) |
| `expected` | `TRUE`, `FALSE` or `UNKNOWN` |
| `z3` | the answer computed independently by Z3; always equal to `expected` |
| `world_relation` | `conflict`, `agree`, `neutral` (invented words), or `unsupported_true` / `unsupported_false` (the rulebook is silent and the claim is true / false in the real world) |
| `category` | the fact table the content comes from, if any |
| `depth` | the number of reasoning steps from the rulebook to the answer |
| `reasoning` | the reasoning steps, in words (for example `modus_ponens`) |
| `support` | ids of the rulebook statements the answer depends on |
| `traps` | ids of the *trap* statements built for an `UNKNOWN` claim: statements that look like the missing link but do not settle the claim |

### Rulebooks (`level1-rulebooks`, `level2-rulebooks`, `level3-rulebooks`)

One row per rulebook: `kb_id`, `level`, and `statements`, a list in the order shown to the model. Each
statement has:

| Field | Meaning |
|---|---|
| `id`, `text` | identifier and English sentence |
| `form` | the logical form family the sentence was written from (for example `fact`, `rule`) |
| `logic` | the formula checked by Z3; empty for statements that carry no logic (for example "Books generally sing.") |
| `role` | `proof` (needed by some claim), `trap`, or `noise` (a distractor) |
| `world` | `conflict`, `agree` or `neutral` |
| `category` | the fact table the statement comes from, if any |

Each folder also holds `manifest.json`: generator and Z3 versions, configuration, seed, fingerprint (SHA-256)
of the fact tables, counts, and the details of the distractor removal.

## Use

```python
from datasets import load_dataset

claims = load_dataset("lib0rius/ByTheBookTest", "level1", split="test")
rulebooks = load_dataset("lib0rius/ByTheBookTest", "level1-rulebooks", split="test")
rulebook = {r["kb_id"]: "\n".join(s["text"] for s in r["statements"]) for r in rulebooks}

item = claims[0]
context = "KNOWLEDGE BASE\n" + rulebook[item["kb_id"]]
question = item["text"]  # answer TRUE, FALSE or UNKNOWN; compare with item["expected"]
```

The English sentences can be read in more than one way, so the test states ten rules of interpretation to
the model with every claim: for example, different names refer to different entities, and "can", "must" and
"does" are different properties. The expected answers assume those rules. The full prompt is in
[`src/bythebooktest/prompt.py`](https://github.com/lib0rius/ByTheBookTest/blob/main/src/bythebooktest/prompt.py).
The repository can ask a model every claim and compute the measures.

The fields `logic`, `z3`, `world_relation`, `support`, `traps` and the statement `role` reveal the answer or
how it was built: they are for analysis, never for the model.

## First results

On this dataset, d1-2B by Liquid AI, a decision model with 1.9 billion parameters, answers every claim
that the short rulebooks settle (198 of 198), but only 25 of the 102 that they leave open. On the long
rulebooks its accuracy falls from 95.5% at one reasoning step to 43.5% at two. Counting only the facts it
knows, it loses 11 points of accuracy when the rulebook contradicts them, and 18 points when the rulebook is
silent about them. One model and one run: these observations are exploratory. Every answer of the model is in
the GitHub repository (`results/`), and the article reports every measure with its 95% interval.

## Contamination

This dataset is public, so it may end up in the training data of future models. A model that has seen it
could look better than it is. For measures that must stay clean, generate a fresh dataset with another seed:

```bash
bythebooktest generate --config configs/level1.toml --out out/level1
```

with a different `seed` in the configuration. The generator builds a new dataset with the same plan and the
same checks. The same seed gives the same dataset, byte for byte.

## Limitations

- **Synthetic, in English.** Rulebooks are lists of statements written from fixed templates, not real
  documents.
- **Formula versus sentence.** Z3 checks the answer against the formula, not the formula against the English
  sentence. The author reviewed every template and a random sample of 60 claims and found no error, which
  bounds the error rate at about 4.9% with 95% confidence (one-sided Clopper–Pearson: 1 − 0.05^(1/60)).
- **Rules of interpretation.** Four of the ten rules go against a common reading of English (for example,
  "Every bird that is not a penguin flies. Tweety is a bird." leaves "Tweety flies" `UNKNOWN`). Results
  depend on stating them to the model.
- **Real-world truth.** The relation with the real world comes from tables of well-known facts curated by the
  author. It describes common knowledge, not every possible context.

## Version

This is **ByTheBookTest v0.1.0**: version 0.1.0 of the protocol and of its generator, seed 42. The protocol
may change over time. Version numbers follow [semantic versioning](https://semver.org/):

- the **first** number changes when the rules of the protocol change (for example the meaning of `UNKNOWN`
  or the rules of interpretation): results are no longer comparable with earlier versions;
- the **second** changes when something is added (for example a new reasoning group): earlier datasets stay
  valid;
- the **third** changes for corrections.

The leading 0 means the protocol is still young and may change. Each version is tagged both here and in the
GitHub repository, and the changes are listed in
[CHANGELOG.md](https://github.com/lib0rius/ByTheBookTest/blob/main/CHANGELOG.md). To load exactly this
version:

```python
claims = load_dataset("lib0rius/ByTheBookTest", "level1", split="test", revision="v0.1.0")
```

Every `manifest.json` records the generator version, the seed, the configuration and the fingerprint of the
fact tables: the same three give the same dataset, byte for byte. When you report results, cite the version
and the seed, for example "ByTheBookTest v0.1.0, seed 42".

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): free use, including commercial use, with credit
to the author. The code of the generator is under Apache 2.0, in the GitHub repository.

## Citation

```bibtex
@misc{martines2026bythebooktest,
  author       = {Martines, Claudio},
  title        = {ByTheBookTest: does a model follow the rulebook, or what it already knows?},
  year         = {2026},
  version      = {0.1.0},
  howpublished = {\url{https://github.com/lib0rius/ByTheBookTest}},
  note         = {Version 0.1.0, seed 42. Dataset: \url{https://huggingface.co/datasets/lib0rius/ByTheBookTest}}
}
```

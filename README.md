# ByTheBookTest

**Does your model follow the document, or what it already knows?**

ByTheBookTest is a protocol to generate test datasets in which a *rulebook* contradicts world knowledge
("A cactus grows under the sea."), and a test program that measures how often a model answers by the
book. Every expected answer is computed and checked by the [Z3](https://github.com/Z3Prover/z3) theorem
prover before any model sees it.

Each test item is a rulebook and a *claim*. The model must answer, using only the rulebook:

| Answer | Meaning |
|---|---|
| `TRUE` | the claim follows from the rulebook |
| `FALSE` | the negation of the claim follows from the rulebook |
| `UNKNOWN` | neither follows: the rulebook is silent |

Rulebooks come in three lengths (about 120, 760 and 2,400 statements) and mix statements that contradict
the real world, statements that agree with it, and statements made of invented words, so that the effect
of world knowledge can be measured against a fair baseline.

- Article, for a general reader: [article/article.md](article/article.md)
- Dataset: [huggingface.co/datasets/lib0rius/ByTheBookTest](https://huggingface.co/datasets/lib0rius/ByTheBookTest)

## Install

Python 3.12 or later.

```bash
git clone https://github.com/lib0rius/ByTheBookTest.git
cd ByTheBookTest
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Asking a model also needs [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` on the
`PATH` and a GGUF model file. The configurations of the article expect `models/d1-2B-Q8_0.gguf`, from
[LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF).

## Use

Everything goes through one command, run from the repository root:

```bash
# generate a dataset (rulebooks, claims and expected answers checked by Z3)
bythebooktest generate --config configs/level1.toml --out out/level1

# make long rulebooks shorter by removing part of their noise, keeping every claim and answer
bythebooktest prune --source out/level3 --noise-ratio 0.2 --out out/level3-noise20

# ask a model every claim (starts llama-server; can be stopped and resumed at any time)
bythebooktest run --config configs/d1-2b.toml

# follow a long command from another terminal
bythebooktest status runs/d1-2b/status.json

# compute the measures
bythebooktest metrics runs/d1-2b
```

`bythebooktest <command> --help` lists the options of each command. The configuration files are
documented in [`src/bythebooktest/config.py`](src/bythebooktest/config.py); unknown keys are rejected.

The same seed gives the same dataset, byte for byte. Because a public dataset may end up in the training
data of future models, generate a fresh one with another seed for measures that must stay clean.

## Repository

| Folder | Content |
|---|---|
| `src/bythebooktest/` | generator, test program, measures |
| `data/tables/` | curated tables of well-known facts, with their false versions |
| `configs/` | the configurations used in the article |
| `dataset/` | the released dataset: Level 1, Level 2 and Level 3 rulebooks with their claims, and the sentences of the knowledge probe (the same files as on Hugging Face) |
| `results/` | every answer of the tested model, the run records and the measures; `bythebooktest metrics results/d1-2b` recomputes them |
| `scripts/` | the script that draws the figures of the article |
| `article/` | the article with its figures |
| `tests/` | automated tests (`pytest`) |

## Licence

Code: [Apache 2.0](LICENSE). Dataset, tables, results and article: [CC BY 4.0](LICENSE-data).

## Versions

The protocol and its generator share one version number, following [semantic versioning](https://semver.org/):
the first number changes when the rules of the protocol change and results stop being comparable, the second
when something is added, the third for corrections. The released dataset is **v0.1.0, seed 42**. Each version
is tagged here and on Hugging Face; [CHANGELOG.md](CHANGELOG.md) lists the changes.

## Citation

Please cite the version you used.

```
Martines, C. (2026). ByTheBookTest: does a model follow the rulebook, or what it already knows?
Version 0.1.0. https://github.com/lib0rius/ByTheBookTest
```

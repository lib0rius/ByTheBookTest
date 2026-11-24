# Changelog

All notable changes to the ByTheBookTest protocol, its generator and the released dataset. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[semantic versioning](https://semver.org/):

- **major** (first number): the rules of the protocol change, for example the meaning of an answer or the rules
  of interpretation; results are not comparable with earlier versions;
- **minor** (second number): something is added, for example a reasoning group or a level; earlier datasets
  stay valid;
- **patch** (third number): corrections that change neither the rules nor the released dataset.

Every entry says whether results stay comparable with the previous version.

## [0.1.0] — unreleased

First public version.

### Protocol

- Three answers, `TRUE`, `FALSE` and `UNKNOWN`, defined by classical logical consequence from the rulebook,
  with unique names and an open world.
- Ten rules of interpretation stated to the model with every claim.
- Expected answers computed by construction and checked independently by Z3 on the complete rulebook; a claim
  is kept only if the two agree.
- Every group of 100 claims has the same mix: conflict with the real world, agreement, invented words, and a
  rulebook silent on a real fact.

### Generator

- Three levels: 3, 9 and 14 reasoning groups, 0–1, 0–3 and 0–6 reasoning steps.
- Distractors, unrelated on Level 2 and similar to useful statements on Level 3, with pruning to a chosen
  ratio.
- Knowledge probe: affirmative and negative sentences for every fact of the tables, with no rulebook.
- Fact tables: 305 well-known facts in five tables and 26 abilities.

### Dataset

- Seed 42: Level 1, Level 2 with 10% of its distractors, Level 3 with 20%; 12 rulebooks, 2,600 claims (858
  `TRUE`, 858 `FALSE`, 884 `UNKNOWN`); knowledge probe of 1,580 sentences.

[0.1.0]: https://github.com/lib0rius/ByTheBookTest/releases/tag/v0.1.0

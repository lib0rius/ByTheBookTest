# The Hidden Conflict in AI Decisions: Introducing ByTheBookTest

![Cover: an orange fish looks at a cactus under the sea. The rulebook says "A cactus grows under the sea."; the
answers are TRUE, FALSE or UNKNOWN.](figures/fig0-cover.png)

*Does your AI follow the document, or what it already knows? An example protocol to generate test datasets
where the document contradicts the model, and a first test on a one-step decision model.*

**Claudio Martines (lib0rius)** · November 2026 · [GitHub](https://github.com/lib0rius/ByTheBookTest) · [dataset on Hugging Face](https://huggingface.co/datasets/lib0rius/ByTheBookTest)

---

## 1. A cactus under the sea

I gave a small AI model a rulebook and a statement, and asked a single question: according to this
rulebook, is the statement TRUE, FALSE or UNKNOWN?

The rulebook lies about the world on purpose. One of its rules says: *"A cactus grows under the sea."*
Asked to judge *"A cactus does not grow under the sea"*, the model answered FALSE, exactly as the rulebook
requires. On a one-page rulebook it did this every time: whenever the rulebook settled a question, the
model found the answer, in 198 cases out of 198, even when the rulebook contradicted the real world.

But a rulebook also leaves most questions open. Another short rulebook mentions Jupiter only once, to say
that it is a "trokulus", an invented word; it says nothing about its size. Asked to judge *"Jupiter is the
smallest planet in the Solar System"*, the only correct answer is UNKNOWN: the rulebook does not say. The
model answered FALSE, which is true in the world but is not what the rulebook says. On the short
rulebooks, when the rulebook left the question open, the model admitted it only one time in four. The other
three times it made up an answer.

On a rulebook sixteen times longer, the picture gets worse. The model found the answer the
rulebook gives less than half the time, and its own knowledge of the world started to leak into its
answers, even on facts that it demonstrably knows.

The cactus and Jupiter are examples built on purpose. In real documents the same two mistakes look more
ordinary, and that makes them harder to notice. A standard updated this year changes a threshold or a rate:
a model that trusts what it learned answers with the value of the previous version. A company travel policy
says nothing about business class on night flights: the right answer is "the policy does not say, a person
decides", but a model that fills the gap with what it knows answers "allowed", because many companies allow
it.

So before trusting an AI decision on your documents, measure whether it follows them. No existing test did
what I needed, so I designed an example protocol to generate one, **ByTheBookTest**: rulebooks that
deliberately contradict what models know, with every expected answer checked independently. I used it to
generate a dataset and test a one-step decision model. Such a model doesn't reason out loud: it weighs your
document against its own training, and you can't see which one won.

## 2. Why this matters

More and more AI systems decide on the basis of documents: internal procedures, contracts, technical
standards, regulations, audit checklists. In all of them the document is the authority: when the
document and general knowledge disagree, the document must win.

When the document says nothing, the right answer is "the document does not say". That is not a failure.
In an audit or a compliance check it is often the most valuable answer, because it marks a gap that a
person has to fill.

These errors are hard to catch in normal testing. On ordinary documents the document and the world usually
agree, so a model that quietly answers from its training looks just as good as one that reads.

Tests of this kind exist, and some are very good: they generate logic problems with three answers, TRUE,
FALSE and UNKNOWN, and check every expected answer with an automatic prover. None of the thirteen I compared,
though, has all the pieces this question needs: a long document shared by many questions,
contents chosen on purpose to contradict well-known facts, and groups of questions that tell the document
apart from the model's own knowledge. None measures what a model does when the document is silent on
something it knows.

## 3. How a one-step decision model answers

Most people now picture an AI model as something that writes: it answers in sentences, and the newer ones
reason step by step before answering. The model I tested does neither. It belongs to a recent class of
*decision models*, whose name recalls the "System 1" of fast, intuitive thinking. You give it a context
(here, the rulebook) and a question with a fixed set of options (TRUE, FALSE, UNKNOWN). In a single pass it
returns a choice and a probability for each option. It writes no text and no reasoning.

This is attractive for real use. It is fast: a fraction of a second per decision on a one-page rulebook. It
is cheap enough to run on a desktop computer.

It also changes what you need to keep in mind when you rely on such a model:

- **How it reasons.** There is no chain of reasoning to read. The decision is a weighing of the options
  inside the network, and the only trace it leaves is a set of probabilities.
- **What it already contains.** Its weights hold what it learned in training. That knowledge takes part in
  every answer, whether or not you ask for it.
- **The hidden conflict.** When the document says one thing and the training says another, the answer is a
  compromise between two sources. From the outside you see only the result, not which source prevailed.

A rulebook that agrees with the world cannot reveal this conflict: the document and the training give the
same answer, and you cannot tell which one the model used. Only a rulebook that contradicts the world can.
That is why the test is built the way it is.

## 4. The protocol: ByTheBookTest

ByTheBookTest is a protocol for generating test datasets in which the document is the only valid source of
truth, and the expected answer to every question is certain. The dataset used here is one instance,
generated with version 0.1.0 of the protocol and a fixed random seed; anyone can generate a fresh one with the
same rules.

![How ByTheBookTest is built and run: fact tables, generator, Z3 check, model under test, measures; the
knowledge test without a rulebook runs on the same fact tables.](figures/fig-protocol.png)

### 4.1 The task

Each test item is a **rulebook**, a list of short English statements, and a **claim**, the statement to
judge. The model must answer:

- **TRUE** if the claim follows logically from the rulebook;
- **FALSE** if the opposite of the claim follows logically from the rulebook;
- **UNKNOWN** if neither follows.

The reference is classical logic, and every valid inference counts, including reasoning backwards: if
every tarv is a pelm and Kiro is not a pelm, then Kiro is not a tarv.

### 4.2 Rulebooks that contradict the world, by design

Many statements in each rulebook deliberately violate very well-known facts, in five areas:

| Area | Example statement in a rulebook |
|---|---|
| History and geography | "Napoleon was born in Australia." |
| Nature | "The Sun rises twice a day." |
| Mathematics and geometry | "One plus one equals three." |
| Common sense | "A dropped stone rises into the sky." |
| Implicit assumptions | "One plus one is written as 10", which is true only in base 2 |

The facts come from hand-written tables of things everyone learns at school, each with its true version
and one or more false ones; the false version goes into the rulebook. Only the content contradicts the
world, never logic itself: the way conclusions are drawn stays normal, otherwise no expected answer would
be certain. And each rulebook is consistent with itself. It may contradict the world, but never itself.

### 4.3 What the answers are compared with

A drop in accuracy on the statements that contradict the world means nothing on its own. Every claim
therefore carries its relation to the real world, and the test compares groups of claims that differ in
that relation only:

| Relation to the real world | Example | Right answer |
|---|---|---|
| The rulebook **contradicts** the world | Rulebook: "A cactus grows under the sea." Claim: "A cactus does not grow under the sea." | FALSE |
| The rulebook **agrees** with the world | Rulebook: "Berlin is the capital of Germany." Claim: "Berlin is not the capital of Germany." | FALSE |
| The rulebook is **silent** on a real fact | Rulebook: "Jupiter is a trokulus." and nothing else about Jupiter. Claim: "Jupiter is the smallest planet in the Solar System." | UNKNOWN |
| **Invented** words, nothing to know | Rulebook: "Kugunut is a kregregek." Claim: "Kugunut is not a kregregek." | FALSE |

From these groups come the two measures of how much the model's own knowledge gets in the way. For both,
0 is the ideal and a negative value means that knowledge interferes:

- the **conflict effect**: accuracy when the rulebook contradicts the world, minus accuracy when it agrees.
  Both groups talk about real things and have the same mix of answers and of reasoning steps, so the
  difference measures only whether the rulebook is on the side of the world;
- the **silence effect**: on real facts the rulebook does not mention, how often the model answers in the
  direction opposite to the real world, minus how often it answers in the direction of the real world. A
  model that reads only the rulebook either says UNKNOWN or errs at random, and scores 0.

The agreeing claims also close a loophole. Without them, a model could score well on the contradicting
claims simply by answering the opposite of what it knows, without reading the rulebook at all. The claims
with invented words show the reasoning when there is nothing to know; the results explain why they are not
the main comparison.

### 4.4 Every expected answer checked independently

The generator builds each claim from a known proof, or, for UNKNOWN, from a proof with a link deliberately
missing. Then an automatic theorem prover, **Z3**, checks every expected answer on its own, in classical
logic, by asking two questions: can the rulebook be true while the claim is false, and can the rulebook be
true while the claim is true? A claim is kept only if the prover and the construction agree; otherwise it is
built again. Every one of the 2,600 claims of the dataset passed, 9 of them after being rebuilt.

### 4.5 Every interpretation choice is written down

The hardest part of building the dataset was not generating sentences. It was making sure that no expected
answer depends on a convention the model would have to guess. Some of the questions that had to be
settled:

- Does "Kiro lives in Dask" mean that Kiro does not live in Morv? (No, unless the rulebook says that a
  person lives in one place only.)
- Does "Birds generally fly" say anything about one particular bird? (No.)
- Does "Every pilot must fly" mean "Every pilot can fly"? (Not unless the rulebook says so, although in
  everyday language it would.)
- Is "Every moon of Zeta is red" true if Zeta has no moons? (Yes, in classical logic.)
- Is the year 2025 included in "from 2020 to 2025"? (Yes.)

Each choice is written in the instructions given to the model, as one of ten rules of interpretation. The
model is judged on rules it has been told, not on conventions it must guess.

### 4.6 Three rulebook lengths

| | Short | Medium | Long |
|---|---:|---:|---:|
| Statements per rulebook | about 120 | about 760 | about 2,400 |
| Length in tokens (the units a model reads) | about 1,600 | about 9,100 | about 26,600 |
| Rulebooks, and claims per rulebook | 4 × 75 | 4 × 225 | 4 × 350 |
| Reasoning steps needed, at most | 1 | 3 | 6 |
| Irrelevant statements ("noise") | none | 10% | 20% |
| Logical forms | facts, simple rules; "generally", "can", "must", "is allowed to" | adds negations, several conditions, "or", "either… or", "if and only if", exceptions, dates | adds relations between entities, "at least one", closed lists, counts, "every" |

A reasoning step is one link in a chain of deductions. "Fubetrir is a gegibux; every gegibux can knit"
takes one step to conclude that Fubetrir can knit. If the rulebook said instead that every gegibux is a
spogrelux and every spogrelux can knit, it would take two.

Each length contains all the forms of the shorter ones. Claims are spread evenly across groups of logical
forms, 100 claims per group and length, so every part of the test has a similar uncertainty. The full
dataset has 2,600 claims across 12 rulebooks.

### 4.7 What made it hard

- **Certainty versus naturalness.** Everyday language implies things that logic does not: "must" suggests
  "can", "generally" suggests "always". Each case had to be settled and written down.
- **Interference inside a rulebook.** In a large rulebook, a statement added for one claim, or as noise,
  can change the answer to another claim. The check is therefore done on the whole rulebook, not claim by
  claim; for the long rulebooks, which are too large for the prover in one piece, on each group of
  statements that are linked to each other.
- **Fair proportions.** Each length has the same mix of right answers (a third TRUE, a third FALSE, a third
  UNKNOWN) and the same mix of relations to the world, so that no answer is favoured by chance.
- **Human review.** A prover checks logic, not English. All sentence templates and a sample of 60 claims
  were reviewed by hand.

## 5. The test

I ran the whole dataset on a single desktop computer, with open-source software only.

| Item | Value |
|---|---|
| Model | LiquidAI **d1-2B**, a decision model with 1.9 billion parameters and a context of 128,000 tokens, in 8-bit form (GGUF Q8_0) |
| Engine | llama.cpp (build b12359), which serves decision models through a dedicated interface |
| Hardware | AMD Ryzen AI Max+ 395 with its integrated Radeon 8060S graphics |
| Data | All 2,600 claims: 300 on the short rulebooks, 900 on the medium ones, 1,400 on the long ones |
| What the model receives | The rulebook as context; then the instructions with the ten rules of interpretation, the claim, and the three options with their definitions |
| Repetitions | One per claim: the model is deterministic, and the same claim asked twice, at different times and under different load, gave identical probabilities to the last digit |
| Time per claim | About 0.3 seconds on a short rulebook, 2 on a medium one, 9 on a long one; about four hours for the whole dataset |

### Does the model know the facts?

A rulebook can only contradict what the model knows. If the model had never learned that the Sun rises in
the east, a rulebook saying it rises in the west would not be in conflict with anything. So, before reading
the results, I checked what the model knows, with a short test of its own: no rulebook, and a different
question.

I took every statement of the hand-written tables, true and false versions alike, each in its positive and
its negative form ("The Sun rises in the east", "The Sun does not rise in the east"): 1,580 sentences in
all. For each one I asked the model whether it is true or false *in the real world*, with UNKNOWN allowed
for "I do not know". A fact counts as known only if the model gets both forms right, so that a model that
always says TRUE does not look well informed by accident.

The model knows **83%** of the facts: almost all of history and geography (98%), most of nature, common
sense and mathematics (82–84%), and just over half of the implicit assumptions (57%). It almost never says
"I do not know": 15 times out of 1,580. The results below use this test to separate the facts the model
knows from those it does not.

## 6. What the test showed

Overall, the model got 74% of the claims right on the short rulebooks, 53% on the medium ones and 41% on
the long ones. Answering at random among three options would give 33%. These averages hide four very
different behaviours.

### 6.1 It finds the answer when there is one, but rarely admits when there is none

Split the claims in two. In the first group the rulebook settles the question, and the right answer is
TRUE or FALSE. In the second the rulebook is silent, and the right answer is UNKNOWN.

| | Short | Medium | Long |
|---|---:|---:|---:|
| **The rulebook gives the answer:** right | **100%** | 68% | 43% |
| … answered the opposite (TRUE for FALSE, or the reverse) | 0% | 23% | 29% |
| … answered UNKNOWN | 0% | 10% | 29% |
| **The rulebook is silent:** answered UNKNOWN | **25%** | 25% | 39% |

The full picture is in Figure 1, which shows one *confusion matrix* for each rulebook length. A confusion
matrix crosses two things: each **row** is the right answer, each **column** is the answer the model gave
(T, F and U stand for TRUE, FALSE and UNKNOWN). Every row adds up to 100%, so a cell says what share of the
claims with that right answer received that answer. In the short rulebooks, the top-left cell reads 100%:
every claim whose right answer was TRUE was answered TRUE.

The **diagonal** holds the right answers (green bars). Every other cell is a mistake (red bars), and its
position tells which kind:

- **T→F and F→T**: the opposite of what the rulebook says;
- **T→U and F→U**: UNKNOWN, although the rulebook gives the answer;
- **U→T and U→F**: an answer made up where the rulebook is silent.

![Three confusion matrices, rows are the right answer and columns the answer given. Short rulebooks: TRUE
row 100, 0, 0; FALSE row 0, 100, 0; UNKNOWN row 43, 32, 25 (percent). Medium: TRUE row 58, 31, 11; FALSE
row 14, 78, 8; UNKNOWN row 34, 42, 25. Long: TRUE row 38, 32, 30; FALSE row 25, 48, 27; UNKNOWN row 29, 32,
39.](figures/fig3-confusion.png)

*Figure 1. Right answer (rows) against the answer given (columns), by rulebook length. Each row adds up to
100%.*

On a short rulebook the model never gets a stated fact wrong. Every one of its mistakes is of a single kind:
the rulebook says nothing, and the model makes up an answer anyway. It answers UNKNOWN only 8 times in 100,
when a third of the claims call for it.

On longer rulebooks the other two mistakes appear. The model starts answering the opposite of what the
rulebook says, and giving up on questions the rulebook does answer. On the long rulebooks the three
mistakes are mixed together, and the answers are close to chance. Its confidence does not warn you: there,
it is about as sure of its wrong answers (0.54 on a scale from 0 to 1) as of its right ones (0.59).

The matrices also show what the summary table hides: the mistakes are not symmetric. On the medium
rulebooks the model leans towards FALSE. Claims whose right answer is FALSE are answered correctly 78% of
the time, those whose right answer is TRUE only 58%; and when the rulebook is silent, it makes up FALSE
more often than TRUE (42% against 34%). On the long rulebooks the lean is weaker, and the answers spread
almost evenly across the three columns.

### 6.2 One step yes, two steps no

The length of the rulebook is not the main reason for the fall. The number of reasoning steps is. On the
long rulebooks, considering only the claims the rulebook settles:

| Reasoning steps | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Right | 86% | 95% | 43% | 34% | 30% | 38% | 28% |

![Right answers by reasoning steps, one panel per rulebook length, on the same scale of steps from 0 to 6.
Short rulebooks: 100% at 0 and 1 steps. Medium: 98%, 79%, 59%, 47% from 0 to 3 steps. Long: 86%, 95%, 43%,
34%, 30%, 38%, 28% from 0 to 6 steps. A light horizontal line marks chance, 33%.](figures/fig2b-steps-panels.png)

*Figure 2. Right answers by number of reasoning steps, one panel per rulebook length, with 95% intervals.
Claims the rulebook settles (TRUE or FALSE). The light horizontal line is chance (33%).*

With one step the model is almost always right. With two it is right less than half the time, and beyond
two it is close to chance.

Length matters too, but much less. Take claims of the same form, needing the same number of steps, and
compare them across lengths:

| Same kind of claim | Short | Medium | Long |
|---|---:|---:|---:|
| A fact written in the rulebook (0 steps) | 100% | 98% | 86% |
| "Can" or "must" (1 step) | 100% | 100% | 95% |
| Seven forms present in medium and long, 2 steps | — | 59% | 48% |
| The same seven forms, 3 steps | — | 47% | 37% |

A rulebook three times longer costs 5 to 12 points on the same question. A second reasoning step costs
about fifty.

Here the steps build on each other, inside a single answer. If two steps were two independent decisions,
each right as often as a single step, the medium rulebooks would give 0.79 × 0.79 = 62% at two steps; the measured value is
59%, close to it. On the long rulebooks the same arithmetic gives 0.955 × 0.955 = 91%, and the measured value is
43%. There, chaining two steps costs far more than multiplying two chances. One caution: on the long
rulebooks the one-step claims are of only two kinds, "can" and "must", so this comparison is an indication,
not a clean measurement.

Figure 3 breaks the results down by kind of reasoning, this time counting every claim of each kind,
including those whose right answer is UNKNOWN. Each cell is a scale from 0 to 100%: the dot is the share of
right answers, the grey band its 95% interval, and the small tick marks chance (33%). When the band crosses
the tick, the result cannot be told apart from guessing.

![Right answers by kind of reasoning and rulebook length. Short rulebooks: facts written in the rulebook
79%, "can", "must", "generally" 66%, chains of rules 78%. Medium: facts 77%, "can"/"must" 66%, chains 44%,
negations 54%, several conditions 58%, "or" 47%, "if and only if" 51%, exceptions 36%, dates 45%. Long:
facts 67%, "can"/"must" 67%, chains 22%, negations 33%, several conditions 55%, "or" 34%, "if and only if"
35%, exceptions 40%, dates 34%, relations 37%, "some" 42%, counts 45%, "every" 32%, chains sharing a link
37%.](figures/fig5-reasoning-map.png)

*Figure 3. Right answers by kind of reasoning and rulebook length, 100 claims per cell. Dot: measured share.
Grey band: 95% interval. Tick: chance (33%).*

On the short rulebooks every kind of reasoning stays clearly above chance. On the medium ones only
exceptions fall back to it. On the long ones just four kinds stay above chance: facts written in the
rulebook, "can" and "must", several conditions joined by "and", and counts. Chains of rules fall below
chance, to 22%: the more links a deduction needs, the less the model can follow it.

### 6.3 When the rulebook contradicts what the model knows

This is the hidden conflict of section 3, and the two measures of section 4.3 put a number on it. For both,
0 is the ideal and a negative value means that the model's knowledge interferes.

| | Short | Medium | Long |
|---|---:|---:|---:|
| **Conflict effect**: the rulebook contradicts the world, against when it agrees | 0 | −6 | **−12** |
| 95% interval, counting claims | [−3.4, +11.4] | [−15.9, +5.0] | [−20.9, −2.5] |
| 95% interval, counting rulebooks | 0 in all four | [−21.9, +9.4] | [−29.5, +5.7] |
| **Silence effect**: the rulebook is silent on a real fact | −1 | −10 | **−15** |
| 95% interval, counting claims | [−21.6, +18.8] | [−22.2, +1.8] | [−23.6, −6.1] |
| 95% interval, counting rulebooks | [−29.7, +26.9] | [−24.3, +3.9] | [−39.8, +10.1] |
| Conflict effect, only facts the model knows | 0 | −9 | **−11** [−21.5, −0.5] |
| Silence effect, only facts the model knows | −20 | −11 | **−18** [−27.5, −7.7] |

*Points of accuracy. In brackets, the 95% interval: the range within which the true value very likely
lies. "Counting claims" treats every claim as one observation; "counting rulebooks" treats each of the four
rulebooks of a length as one observation (Student t with 3 degrees of freedom), the cautious choice when
claims of the same rulebook may not be independent. The rows on known facts count claims.*

![Right answers when the rulebook agrees with the world and when it contradicts it. Short rulebooks: 100%
and 100%. Medium: 72% and 66%. Long: 56% and 44%. Long rulebooks, only facts the model knows: 49% and
38%.](figures/fig4-conflict.png)

*Figure 4. Right answers when the rulebook agrees with the world (hollow dot) and when it contradicts it
(filled dot). The red segment is the loss, the conflict effect. Last row: long rulebooks, only the facts
the model knows (section 5).*

On the short rulebooks the conflict cannot be seen: the model gets every stated fact right, whether or not
the rulebook agrees with the world, so there is no room for a drop. The −20 of the silence effect on known
facts rests on only 41 claims and is very uncertain (from −44 to +5). On the long rulebooks both measures are
negative, and they stay negative on the facts the model demonstrably knows.

How sure that is depends on how the uncertainty is counted. Counting claims, both intervals on the long
rulebooks exclude zero. Counting rulebooks, they include it: there are only four rulebooks per length, and
each has few claims for each effect (35 agreeing claims per long rulebook). The direction, though, repeats.
The conflict effect is negative in all four long rulebooks (−28, −12, −4 and −4 points), which random signs
would give once in sixteen; the silence effect in three of four (−8, +1, −17 and −36). A separate check
finds that the four rulebooks differ no more than claims built from the same specification do, so counting
claims does not make the intervals too narrow for that reason; counting rulebooks gives the cautious bound.
The direction of the two effects is consistent; their size is uncertain. When the rulebook
contradicts something it knows, the model gets it wrong more often. When the rulebook says nothing about
something it knows, it fills the gap with what it knows.

The two measures look at different claims and point the same way.

There is also a lesson in method here. The obvious baseline for a conflict is a claim with nothing to know,
built from invented words like "Kugunut is a kregregek". Measured that way, the conflict on long rulebooks
would come out at **+9.6**: as if contradicting the world *helped* the model. That comparison mixes two
things, and the numbers show how much of each.

On the long rulebooks, among the claims the rulebook settles, the model gets right 55.7% of those that agree
with the world (78 of 140), 43.8% of those that contradict it (221 of 504) and 34.3% of those made of
invented words (96 of 280). The three groups have the same mix of reasoning steps and logical forms, half
TRUE and half FALSE, so with ordinary words and no knowledge at play they would score the same. The 21.4
points between agreeing and invented claims can come from two sources: knowledge helps when the rulebook
agrees with it, and invented words may cost something in themselves. The test cannot weigh the two
exactly, but if the two effects simply add up it can bound them:

- Knowledge can only push a contradicting claim towards the wrong answer. Yet the contradicting claims
  still beat the invented ones by 9.6 points (43.8% against 34.3%). So the invented words cost at least those
  9.6 points, and more if knowledge also does harm.
- The help of knowledge can then be at most the 11.9 points between agreeing and contradicting claims
  (55.7% against 43.8%), and all of it only if knowledge does no harm to the contradicting ones.

The 95% interval of the 9.6 runs from +2.4 to +16.4, so with the uncertainty of the measure the minimum cost
of the invented words drops to about 2 points: the direction is certain, the size is not.

The cost comes mostly from giving up. On the long rulebooks the model answers UNKNOWN to 46.8% of the
invented-word claims (131 of 280), more than twice as often as to claims about real things (20.4% when the
rulebook contradicts the world, 21.4% when it agrees), and it gives the opposite answer less often (18.9%
against 35.7% and 22.9%). With invented words it more often fails to find the chain of reasoning and says
that the rulebook does not settle the claim. On the medium
rulebooks the three groups are close (72.2%, 66.0%, 68.9%) and show no such cost; on the short ones every
claim of the three groups is right. Differences are computed from unrounded values.

Comparing real against invented words therefore measures partly the words, not the conflict. Only a
comparison between claims that all talk about real things isolates it.

### 6.4 "Can" becomes "does"

Some claims turn on the difference between what something *can* do and what it *does*:

> Rulebook: *"Gikrustok is a plane. Every plane can fly."*
> Claim: *"Gikrustok flies."*

That Gikrustok can fly does not mean that it flies; it may be parked in a hangar. The right answer is
UNKNOWN, and the instructions given to the model say so explicitly. The same holds for "Birds generally
fly. Tweety is a bird.", which says nothing about Tweety. Each length has 34 claims of this kind, with
"can", "must", "is allowed to" or "generally" (17 of the 34 on the short rulebooks use "generally"). The
model gets 0 of them right on the short and medium rulebooks and 4 on the long ones. On the short rulebooks
it answers TRUE to every positive claim and FALSE to every negative one, 34 times out of 34: it reads "can"
as "does" and "generally" as "always", consistently, against its instructions. When the question really is about "can" ("Fubetrir can knit", from "Every gegibux can
knit"), it is right 92% to 100% of the time.

These claims weigh on the short rulebooks: without them, the model gets 84% of the claims right instead of
74%, and admits silence 37% of the time instead of 25%. They stay in the main score, because they were part
of the test before the results were known; both numbers are reported.

One of these conventions can be disputed. If the rulebook says *"Every spogrelux must whistle"*, the test
expects UNKNOWN for *"Dodropot cannot whistle"*, because "must" and "can" are different properties. A
well-known philosophical principle, "ought implies can", would say FALSE, which is the model's answer.
There are three such claims per length.

## 7. What it means

On a one-page document, this small model does what you would hope for: it follows the document, even when
the document is wrong about the world. Three things change that picture, and each suggests something to do
before trusting a decision model with your own documents.

**Silence is the first blind spot.** The model almost never says "the document does not say". In an audit
or a compliance check that is the answer you most need, and this model gives it once in four times, even
on a single page. So measure it. Any test of a document-based AI should include questions that the document
does not answer, and count how often the model admits it.

**Reasoning steps cost more than length.** One step is reliable; two are not. If a decision needs a chain
of deductions, it may be safer to split it into single-step decisions and chain them yourself, where each
link can be checked. Length costs too, 5 to 12 points on the same question: giving the model only the
part of the document it needs may reduce errors. These are suggestions this test does not prove: they are
the next things to measure.

**On long documents, the model's knowledge leaks in, silently.** When the document contradicts something
the model knows, it errs more often; when the document says nothing, it answers from what it knows. Its
confidence does not tell you when this happens. Testing only on documents that agree with the world will
not show it, because there the document and the model's knowledge give the same answer. My reading, stated
as an interpretation and not as a measurement, is that this model weighs the document and its training
together: when the evidence in the document is short and clear, the document wins; when it is
spread across a long text and several steps, the training wins more often.

### Before trusting a decision model with your documents

Four practical points. For each one I say whether it comes from a measure of this test or is a suggestion
still to be checked.

- **Test it on variants of your own documents.** *This is what this test does.* Change a value of the
  document on purpose (a threshold, a date, a definition) and check that the answer changes; add questions
  the document does not answer and count how often the model admits it. These are the two situations that a
  test on ordinary documents cannot see.
- **Do not use the model's confidence as a filter.** *Measured.* On the long rulebooks the model is about as
  sure of its wrong answers (0.54) as of its right ones (0.59). Passing only the least confident answers to a
  person would pass on errors only slightly more often than right answers.
- **Ask the same question with and without the document.** *Suggestion.* A decision model gives the
  probabilities of the options in both cases, at almost no cost. If the answer stays the same once the
  document is removed, the document did not weigh. That is fine when the document and the world say the same
  thing; on a rule that departs from common practice it is a warning sign.
- **Check rules with "can" and "generally" separately.** *Measured.* The model reads "can" as "does" and
  "generally" as "always": on the short rulebooks it did so 34 times out of 34, against its instructions.
  Rules written this way should be checked by hand, or rewritten so that they allow only one reading.

### Limits

This is one test of one small model. It decides in a single step and does not reason before answering, so
the results say nothing about models that do. It ran in one compressed form, with one wording of the
instructions. The rulebooks are synthetic, built from simple sentence templates. The longest is about
27,000 tokens; the model accepts 128,000, and its maker gives 64,000 as the reliable limit. Yet on the same
questions the long rulebooks already cost 5 to 12 points against the medium ones, about three times shorter
(section 6.2), and studies of other models find that the length a model actually uses well is often far
shorter than the declared one, especially when the task needs inference. From short to long rulebooks several things change
at once, not only the length. And while the model knows most of the facts involved, the test cannot tell
whether its interference is limited to the facts it knows. The full list, with the weight of each limit and how it was handled, is
in Appendix B.

### Use the protocol

ByTheBookTest is open. The code is released under the Apache 2.0 licence and the dataset under CC BY 4.0;
you can generate a new dataset with your own seed, or write tables of facts from your own field and test the
models you plan to use. The protocol may evolve: every version has its own number, and this article uses
version 0.1.0. [GitHub](https://github.com/lib0rius/ByTheBookTest) · [dataset on Hugging Face](https://huggingface.co/datasets/lib0rius/ByTheBookTest)

---

## Appendix A. Full measures

All numbers come from the saved answers through the project's measurement program (Appendix C). ↑ means
higher is better, ↓ lower is better. In brackets, 95% intervals: Wilson for proportions, Newcombe for
differences of proportions, normal approximation for the silence effect.

**A.1 The three answers**

| | Short | Medium | Long |
|---|---|---|---|
| Accuracy ↑ | 74.3% [69.1, 78.9] | 53.1% [49.8, 56.4] | 41.4% [38.9, 44.0] |
| Accuracy expected by chance, with the same answer shares | 33.1% | 33.1% | 33.3% |
| Cohen's kappa ↑ (0 = chance, 1 = perfect) | 0.616 | 0.299 | 0.122 |
| Macro-F1 ↑ | 0.690 | 0.507 | 0.413 |
| Answers TRUE / FALSE / UNKNOWN (expected: 33 / 33 / 34) | 48 / 44 / 8% | 35 / 50 / 15% | 31 / 37 / 32% |
| Recall ↑ TRUE / FALSE / UNKNOWN | 100 / 100 / 25% | 58 / 78 / 25% | 38 / 48 / 39% |
| Precision ↑ TRUE / FALSE / UNKNOWN | 69 / 75 / 100% | 54 / 51 / 57% | 40 / 42 / 41% |

**A.2 Confusion matrices** (rows: right answer; columns: answer given, TRUE / FALSE / UNKNOWN)

| Right answer | Short | Medium | Long |
|---|---|---|---|
| TRUE | 99 / 0 / 0 | 172 / 93 / 32 | 174 / 148 / 140 |
| FALSE | 0 / 99 / 0 | 41 / 231 / 25 | 117 / 221 / 124 |
| UNKNOWN | 44 / 33 / 25 | 103 / 128 / 75 | 140 / 151 / 185 |

**A.3 Confidence**

| | Short | Medium | Long |
|---|---:|---:|---:|
| Brier skill score ↑ (1 = perfect, 0 = uniform probabilities) | 0.415 | 0.131 | −0.023 |
| Brier score ↓ | 0.390 | 0.580 | 0.682 |
| Expected calibration error, 10 bins ↓ | 0.128 | 0.108 | 0.144 |
| Mean confidence, right / wrong answers | 0.90 / 0.79 | 0.69 / 0.58 | 0.59 / 0.54 |

**A.4 Accuracy per rulebook ↑**

| Short | Medium | Long |
|---|---|---|
| 70.7, 77.3, 74.7, 74.7% | 52.4, 52.9, 52.9, 54.2% | 41.1, 41.4, 42.0, 41.1% |

**A.5 Knowledge measures** (points of accuracy; 0 is the ideal)

| | Short | Medium | Long |
|---|---|---|---|
| Accuracy: contradicts / agrees / invented words (TRUE or FALSE claims) | 100 / 100 / 100% | 66.0 / 72.2 / 68.9% | 43.8 / 55.7 / 34.3% |
| Conflict effect ↑ | +0.0 [−3.4, +11.4] | −6.2 [−15.9, +5.0] | −11.9 [−20.9, −2.5] |
| Silence effect ↑ | −1.4 [−21.6, +18.8] | −10.2 [−22.2, +1.8] | −14.9 [−23.6, −6.1] |
| Contradicting against invented words (not a measure of conflict, see 6.3) | +0.0 [−3.4, +6.0] | −2.8 [−11.1, +5.8] | +9.6 [+2.4, +16.4] |

**A.6 Facts the model knows and does not know** (test without rulebook, section 5)

| | Short | Medium | Long |
|---|---|---|---|
| Claims about the world linked to a tested fact | 140 of 210 | 560 of 630 | 897 of 980 |
| … of which the model knows the fact | 80.0% | 84.6% | 82.8% |
| Conflict effect, known / not known | +0.0 / +0.0 | −9.4 / +12.9 | −11.0 [−21.5, −0.5] / −26.2 [−42.4, −2.1] |
| Silence effect, known / not known | −19.5 [−44.1, +5.1] / +42.9 | −11.1 / −2.6 | −17.6 [−27.5, −7.7] / −17.9 [−40.1, +4.3] |
| Difference known − not known, conflict / silence | not meaningful (1 and 7 claims) | −22.2 / −8.5 | +15.1 [−7.6, +37.8] / +0.3 [−24.0, +24.5] |

Before the test I wrote down two hypotheses: on long rulebooks the interference would be stronger on known
facts than on unknown ones, and on short rulebooks the silence effect on known facts would stay close to
zero. Neither is confirmed. Most of the "not known" facts are half known (the model gets one of the two
forms right, 99 of 134), because the model almost never answers UNKNOWN, so the test cannot isolate facts
it truly ignores. Claims about invented entities ("Gikrustok flies") cannot be linked to a fact and are
left out.

**A.7 Accuracy by reasoning steps ↑** (claims the rulebook settles)

| Steps | Short | Medium | Long |
|---:|---|---|---|
| 0 | 100% [94.5, 100] | 98.5% [91.9, 99.7] | 86.4% [76.1, 92.7] |
| 1 | 100% [97.2, 100] | 78.8% [73.0, 83.7] | 95.5% [87.5, 98.4] |
| 2 | | 59.3% [51.3, 66.9] | 43.5% [36.1, 51.2] |
| 3 | | 47.4% [39.8, 55.2] | 33.8% [26.9, 41.4] |
| 4 | | | 30.2% [23.6, 37.7] |
| 5 | | | 38.1% [30.8, 45.9] |
| 6 | | | 28.0% [21.6, 35.5] |

**A.8 Accuracy by group of logical forms ↑** (100 claims per cell, about ±10 points)

| Group | Short | Medium | Long |
|---|---:|---:|---:|
| Facts written in the rulebook | 79% | 77% | 67% |
| "Can", "must", "is allowed to", "generally" | 66% | 66% | 67% |
| Chains of rules | 78% | 44% | 22% |
| Negations | | 54% | 33% |
| Several conditions ("and") | | 58% | 55% |
| "Or", "either… or" | | 47% | 34% |
| "If and only if" | | 51% | 35% |
| Exceptions | | 36% | 40% |
| Dates and intervals | | 45% | 34% |
| Relations between entities | | | 37% |
| "Some" | | | 42% |
| Counts ("exactly", "at least") | | | 45% |
| "Every" over a group | | | 32% |
| Chains sharing a link | | | 37% |

**A.9 Without the "can" and "does" claims** (section 6.4)

| | Short | Medium | Long |
|---|---|---|---|
| Accuracy ↑ | 83.8% [78.9, 87.8] | 55.2% [51.9, 58.5] | 42.2% [39.6, 44.8] |
| UNKNOWN recall ↑ | 36.8% | 27.6% | 41.0% |
| "Can"/"does" UNKNOWN claims right ↑ | 0 of 34 | 0 of 34 | 4 of 34 |

## Appendix B. Limits in detail

| Question a reader may ask | Limit | Weight | How it is handled |
|---|---|---|---|
| Does it hold for other models? | One model (1.9 billion parameters), one compressed form (8-bit) | high | Stated with the results |
| | A one-step decision model, not a model that reasons before answering | high | Stated; the comparison is a next step |
| | One wording of the instructions; sensitivity to wording not measured | medium | Stated |
| | "Long" is measured by the loss on the same questions, not by the declared context: about 27,000 tokens against 128,000 accepted and 64,000 given as reliable by the maker, yet 5 to 12 points lost against the medium rulebooks | medium | Stated with the numbers (6.2); lengths named with their size |
| Do the questions measure what they claim? | No knowledge filter when the data were built. Reduced by the test without rulebook: the model knows 83% of the facts, and on long rulebooks both effects stay negative on known facts. Whether interference is limited to known facts stays open (A.6) | medium | Stated with the numbers, including the two hypotheses not confirmed |
| | From short to long, four things change together: length, noise, logical forms, reasoning steps | medium | Two readings at equal conditions (6.2) |
| | Length and noise always grow together; the amount of noise was chosen so that the longest rulebook is processed in one block on the graphics card, not for a reason of measurement | medium | Stated |
| | Ceiling on short rulebooks: every stated fact is right, so the conflict effect cannot show there | medium | Stated next to the number |
| | Few agreeing claims (30, 90, 140 per length): wide intervals for the conflict effect | medium | Intervals reported |
| | Invented words cost something in themselves on the long rulebooks: at least 9.6 points, about 2 with the uncertainty | low | Bounded in 6.3; the conflict is measured only between claims about real things |
| | The "can" and "does" convention, and the "ought implies can" case | low | Kept in the score; reported separately (6.4, A.9) |
| | "If and only if" rules begin with "someone" or "something" ("Someone is a kelvar if and only if that person was born in Australia"): meant as "anyone", as in a definition, but it can be read as "there is someone who" | low | Stated; the data are unchanged |
| | Synthetic data from sentence templates | medium | Stated |
| How solid are the numbers? | Claims of the same rulebook are not independent; accuracy per rulebook is very similar (long: 41.1–42.0%) | medium | Stated with the numbers (A.4) |
| | Many comparisons, no correction for multiple testing; no plan written before the main test | medium | Stated. Fixed before the test: accuracy, precision and recall per answer, accuracy by number of steps and by kind of reasoning, a conflict comparison and the knowledge pull on silent claims. Set after the first results: the conflict effect against agreeing claims (claims with invented words measure the words too) and without UNKNOWN claims (they measure abstention too), and the silence effect as the difference between the two directions. The four observations are exploratory; the hypotheses of the test without a rulebook were written before it ran |
| | On long rulebooks the 0- and 1-step claims are of only two kinds | low | Stated (6.2) |
| | In the test without rulebook, some groups on short rulebooks have 1 to 7 claims; the minimum was set on the total, not on each measure | low | Stated (A.6) |
| Was the test run correctly? | 24 claims of the long rulebooks timed out because the inference server stalled, and were repeated with one slot | low | The model is deterministic: 12 claims asked twice, under different load, gave identical probabilities |

## Appendix C. How to reproduce

| Item | Value |
|---|---|
| Model file | `d1-2B-Q8_0.gguf` from `LiquidAI/d1-2B-GGUF`, 1,846,240,576 bytes, SHA-256 `073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c` |
| Engine | llama.cpp build b12359, `llama-server`, Vulkan, options `-b 32768 -ub 32768 -fa on --cache-ram 8192`, context 128,000, 4 slots |
| Fact tables | SHA-256 `494382072eeb8976e41c48b2065d4984a1a7b0dd157005e747a15ccb48b57776` |
| Instructions (system prompt and template) | SHA-256 `e505f514e7ca67d331a4dec35cb8023eec9b11a605ef4bf1fc1431ed8e19e27f` |
| Dataset of the main test | ByTheBookTest v0.1.0, seed 42; SHA-256 of rulebooks and claims `de62497b27dae39926c24acce2525ca33c4e8fe797b45a802ec6bf1995cbf0e7` |
| Dataset of the test without rulebook | SHA-256 `79331110b49d0fedb85eff346d1cef94189eb4f226cb1f468dd44173fdae36b7` |

The rulebooks were generated with more noise (half of the statements on medium rulebooks, three quarters on
long ones) and then thinned to 10% and 20% by removing noise statements only, at random with a fixed seed;
every claim and every expected answer was checked again with Z3 after thinning.

The commands below run from the root of the repository. The released dataset is in `dataset/`, and the
generation commands rebuild the same files in `out/`; the test configurations read `dataset/` and expect
the model file in `models/`.

```bash
# dataset
bythebooktest generate --config configs/level1.toml --out out/level1
bythebooktest generate --config configs/level2.toml --out out/level2
bythebooktest generate --config configs/level3.toml --out out/level3
bythebooktest prune --source out/level2 --noise-ratio 0.1 --out out/level2-noise10
bythebooktest prune --source out/level3 --noise-ratio 0.2 --out out/level3-noise20

# main test, the repetition of timed-out claims, measures
bythebooktest run --config configs/d1-2b.toml
bythebooktest run --config configs/d1-2b.toml --retry-timeouts --parallel 1
bythebooktest metrics runs/d1-2b

# test without rulebook
bythebooktest probe build --out out/world-probe
bythebooktest run --config configs/d1-2b-world.toml
bythebooktest probe analyze runs/d1-2b-world runs/d1-2b
```

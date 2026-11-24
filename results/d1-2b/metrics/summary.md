# Run d1-2b: main measures

Lenient reading of the answers; the exact reading also measures compliance with the answer format. In brackets, the 95% confidence interval. The conflict effect compares claims that contradict the real world with claims that agree with it; the silence effect, on UNKNOWN claims about real facts, is the share of answers against the real world minus the share in its direction. For both, 0 is ideal and a negative value means that world knowledge interferes. The invented-words effect compares conflict claims with TRUE or FALSE claims made of invented words.

| Model | Level | Claims | Accuracy | Conflict effect | Silence effect | Invented-words effect | Intrusion (TRUE) | Intrusion (FALSE) | Accuracy when agreeing | Invalid | Thinking budget hit | Exact accuracy | Invalid (exact) |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| d1-2B | 1 | 300 | 74.3% [69.1%, 78.9%] | +0.0% [-3.4%, +11.4%] | -1.4% [-21.6%, +18.8%] | +0.0% [-3.4%, +6.0%] | 47.2% [32.0%, 63.0%] | 30.6% [18.0%, 46.9%] | 100.0% [88.6%, 100.0%] | 0.0% [0.0%, 1.3%] | 0.0% [0.0%, 1.3%] | 74.3% [69.1%, 78.9%] | 0.0% [0.0%, 1.3%] |
| d1-2B | 2 | 900 | 53.1% [49.8%, 56.4%] | -6.2% [-15.9%, +5.0%] | -10.2% [-22.2%, +1.8%] | -2.8% [-11.1%, +5.8%] | 43.5% [34.6%, 52.9%] | 49.1% [39.8%, 58.4%] | 72.2% [62.2%, 80.4%] | 0.0% [0.0%, 0.4%] | 0.0% [0.0%, 0.4%] | 53.1% [49.8%, 56.4%] | 0.0% [0.0%, 0.4%] |
| d1-2B | 3 | 1400 | 41.4% [38.9%, 44.0%] | -11.9% [-20.9%, -2.5%] | -14.9% [-23.6%, -6.1%] | +9.6% [+2.4%, +16.4%] | 43.5% [36.2%, 51.0%] | 40.5% [33.3%, 48.0%] | 55.7% [47.4%, 63.7%] | 0.0% [0.0%, 0.3%] | 0.0% [0.0%, 0.3%] | 41.4% [38.9%, 44.0%] | 0.0% [0.0%, 0.3%] |

## Without the modal UNKNOWN claims

| Model | Level | Claims | Accuracy | UNKNOWN recall | Modal UNKNOWN claims right |
|---|---:|---:|---|---|---|
| d1-2B | 1 | 266 | 83.8% [78.9%, 87.8%] | 36.8% [26.3%, 48.6%] | 0/34 |
| d1-2B | 2 | 866 | 55.2% [51.9%, 58.5%] | 27.6% [22.6%, 33.2%] | 0/34 |
| d1-2B | 3 | 1366 | 42.2% [39.6%, 44.8%] | 41.0% [36.5%, 45.6%] | 4/34 |

## Same text, more steps (TRUE or FALSE claims)

| Model | Level | Depth | Accuracy |
|---|---:|---:|---|
| d1-2B | 1 | 0 | 100.0% [94.5%, 100.0%] |
| d1-2B | 1 | 1 | 100.0% [97.2%, 100.0%] |
| d1-2B | 2 | 0 | 98.5% [91.9%, 99.7%] |
| d1-2B | 2 | 1 | 78.8% [73.0%, 83.7%] |
| d1-2B | 2 | 2 | 59.3% [51.3%, 66.9%] |
| d1-2B | 2 | 3 | 47.4% [39.8%, 55.2%] |
| d1-2B | 3 | 0 | 86.4% [76.1%, 92.7%] |
| d1-2B | 3 | 1 | 95.5% [87.5%, 98.4%] |
| d1-2B | 3 | 2 | 43.5% [36.1%, 51.2%] |
| d1-2B | 3 | 3 | 33.8% [26.9%, 41.4%] |
| d1-2B | 3 | 4 | 30.2% [23.6%, 37.7%] |
| d1-2B | 3 | 5 | 38.1% [30.8%, 45.9%] |
| d1-2B | 3 | 6 | 28.0% [21.6%, 35.5%] |

## Same question, longer text (TRUE or FALSE claims)

| Model | Depth | Groups | Level | Accuracy |
|---|---:|---|---:|---|
| d1-2B | 0 | direct_fact | 1 | 100.0% [94.5%, 100.0%] |
| d1-2B | 0 | direct_fact | 2 | 98.5% [91.9%, 99.7%] |
| d1-2B | 0 | direct_fact | 3 | 86.4% [76.1%, 92.7%] |
| d1-2B | 1 | modal | 1 | 100.0% [94.5%, 100.0%] |
| d1-2B | 1 | modal | 2 | 100.0% [94.5%, 100.0%] |
| d1-2B | 1 | modal | 3 | 95.5% [87.5%, 98.4%] |
| d1-2B | 2 | conjunction, disjunction, exception, iff, negation, rule_chain, time | 2 | 59.3% [51.3%, 66.9%] |
| d1-2B | 2 | conjunction, disjunction, exception, iff, negation, rule_chain, time | 3 | 47.9% [38.1%, 57.9%] |
| d1-2B | 3 | conjunction, disjunction, exception, iff, negation, rule_chain, time | 2 | 47.4% [39.8%, 55.2%] |
| d1-2B | 3 | conjunction, disjunction, exception, iff, negation, rule_chain, time | 3 | 36.6% [27.5%, 46.7%] |

## The three answers

Higher is better. Answerable claims: accuracy on TRUE or FALSE claims, and how they split into right / inverted (TRUE instead of FALSE or the other way round) / abstained (UNKNOWN). UNKNOWN recall: how often the model admits that the rulebook is silent.

| Model | Level | Accuracy | Chance | Kappa | Answers T / F / U | Answerable claims | Right / inverted / abstained | UNKNOWN recall | Recall T / F / U | Precision T / F / U | Macro-F1 |
|---|---:|---|---:|---:|---|---|---|---|---|---|---:|
| d1-2B | 1 | 74.3% [69.1%, 78.9%] | 33.1% | 0.616 | 48% / 44% / 8% | 100.0% [98.1%, 100.0%] | 100% / 0% / 0% | 24.5% [17.2%, 33.7%] | 100% / 100% / 25% | 69% / 75% / 100% | 0.690 |
| d1-2B | 2 | 53.1% [49.8%, 56.4%] | 33.1% | 0.299 | 35% / 50% / 15% | 67.8% [64.0%, 71.5%] | 68% / 23% / 10% | 24.5% [20.0%, 29.6%] | 58% / 78% / 25% | 54% / 51% / 57% | 0.507 |
| d1-2B | 3 | 41.4% [38.9%, 44.0%] | 33.3% | 0.122 | 31% / 37% / 32% | 42.7% [39.6%, 46.0%] | 43% / 29% / 29% | 38.9% [34.6%, 43.3%] | 38% / 48% / 39% | 40% / 42% / 41% | 0.413 |

## Confidence and accuracy per rulebook

| Model | Level | Brier skill score ↑ | Brier ↓ | ECE ↓ | Confidence, right / wrong | Accuracy per rulebook ↑ |
|---|---:|---:|---:|---:|---|---|
| d1-2B | 1 | 0.415 | 0.390 | 0.128 | 0.90 / 0.79 | L1-KB1 70.7%, L1-KB2 77.3%, L1-KB3 74.7%, L1-KB4 74.7% |
| d1-2B | 2 | 0.131 | 0.580 | 0.108 | 0.69 / 0.58 | L2-KB1 52.4%, L2-KB2 52.9%, L2-KB3 52.9%, L2-KB4 54.2% |
| d1-2B | 3 | -0.023 | 0.682 | 0.144 | 0.59 / 0.54 | L3-KB1 41.1%, L3-KB2 41.4%, L3-KB3 42.0%, L3-KB4 41.1% |

## Overall, by model

Accuracy on the claims as they are (Level 3 weighs more) and with the same weight for every level.

| Model | Claims | Answers | Accuracy | Accuracy, levels weighted equally | Conflict effect |
|---|---:|---:|---|---|---|
| d1-2B | 2600 | 2600 | 49.3% [47.4%, 51.2%] | 56.3% [54.1%, 58.4%] | -8.5% [-14.8%, -1.8%] |

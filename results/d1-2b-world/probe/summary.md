# Knowledge probe: d1-2b-world with d1-2b

A sentence pair (affirmative and negative) is known if the model is right on both. Effects: 0 is ideal, negative when world knowledge interferes.

## d1-2B

- Known pairs: 83.0% [80.3%, 85.5%] (656/790)
- Right sentences: 89.3% [87.7%, 90.7%] (1411/1580)
- Answers TRUE / FALSE / UNKNOWN: 698 / 867 / 15

| Table / category | Known pairs |
|---|---|
| world_ability/common_sense | 100.0% [75.8%, 100.0%] (12/12) |
| world_ability/nature | 92.9% [68.5%, 98.7%] (13/14) |
| world_fact/common_sense | 82.8% [75.6%, 88.3%] (111/134) |
| world_fact/history_geography | 98.5% [95.6%, 99.5%] (192/195) |
| world_fact/implicit_assumptions | 56.6% [47.7%, 65.0%] (69/122) |
| world_fact/math_geometry | 81.9% [74.9%, 87.2%] (122/149) |
| world_fact/nature | 83.5% [77.1%, 88.4%] (137/164) |

| Level | Real-world claims | Linked | Known share | Group | Conflict effect | Silence effect |
|---:|---:|---:|---|---|---|---|
| 1 | 210 | 140 | 80.0% [72.6%, 85.8%] (112/140) | known (112) | +0.0% [-6.9%, +16.8%] | -19.5% [-44.1%, +5.1%] |
| 1 | 210 | 140 | 80.0% [72.6%, 85.8%] (112/140) | unknown (28) | +0.0% [-16.1%, +79.3%] | +42.9% [+6.2%, +79.5%] |
| 1 | | | | difference | +0.0% [-49.2%, +49.2%] | -62.4% [-106.5%, -18.2%] |
| 2 | 630 | 560 | 84.6% [81.4%, 87.4%] (474/560) | known (474) | -9.4% [-20.6%, +3.4%] | -11.1% [-25.2%, +3.0%] |
| 2 | 630 | 560 | 84.6% [81.4%, 87.4%] (474/560) | unknown (86) | +12.9% [-18.3%, +47.1%] | -2.6% [-30.5%, +25.4%] |
| 2 | | | | difference | -22.2% [-57.0%, +12.6%] | -8.5% [-39.9%, +22.8%] |
| 3 | 980 | 897 | 82.8% [80.2%, 85.2%] (743/897) | known (743) | -11.0% [-21.5%, -0.5%] | -17.6% [-27.5%, -7.7%] |
| 3 | 980 | 897 | 82.8% [80.2%, 85.2%] (743/897) | unknown (154) | -26.2% [-42.4%, -2.1%] | -17.9% [-40.1%, +4.3%] |
| 3 | | | | difference | +15.1% [-7.6%, +37.8%] | +0.3% [-24.0%, +24.5%] |

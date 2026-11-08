# ccbench v1: clean utility and manipulation sensitivity of daily topic selection

Every day at a fixed cutoff, a monitor picks k = 3 topics of a domain. The score is the number of distinct accounts
that first engage with any of the three topics in the 24 h after the cutoff, as a share of the best possible pick.
Each method is scored on the same decisions twice: on clean data, and under a fixed suite of simulated
manipulation attacks. In each attack, an attacker lifts one low-yield victim topic per day with fake accounts.
Fake accounts never count as genuine new responders, so the target stays unchanged.

## Data (`outputs/ccbench_v1/`, built by `scripts/www_bench_export.py`)

| tier | settings | collected |
|---|---|---|
| dev | cyber_main, cyber_fresh, nba, crypto, ai | before method design |
| held | climate, film, games, soccer, space, stocks | after the first pre-registration (DOSE2) |
| test | music, books, f1, tennis, health, food | after all methods and tests were frozen (DOSE3) |

- **Source:** public Bluesky search results with English text, about 10,000 random root posts per setting, plus
  their reply and quote neighbourhoods.
- **Topics:** the matched search terms or topic labels, about 14 per domain.
- **Days:** each setting has 3 history days (calibration), 4 development days (training) and 8 evaluation days.
  cyber_main is the exception, with 7 / 12 / 12.
- **File contents:** one JSON per setting. Each topic-day cell lists:
  - its posts as (author, replies before the cutoff);
  - its responders as {account: replies};
  - conversational responders;
  - which depth-1 repliers were answered, and by whom;
  - unanswered depth-1 repliers;
  - the target set of next-day new responders.
- **Privacy:** accounts are pseudonymous integers, namely the rank of the internal account id in one global sorted
  list. The files contain no DIDs, handles, text or URIs. The data are derived from public posts, so they are
  pseudonymous, not anonymous.

## Protocol

- **Calibration:** a method may calibrate on the setting's history days.
- **Learned methods:** they train on the development days of the dev tier. Each dev setting is evaluated with its
  own fold left out; the two cyber windows form one fold.
- **Attacks:**
  - **A1–A11 (partial):** reply bombs, Sybil swarms, aged and compromised accounts, pods, sockpuppets, brigades,
    staged conversation, replies to real replies, post floods, and post floods with replies.
  - **A12/A13 (all sides):** posts, replies and staged answers combined.
  - **Settings:** budgets are taken from measured campaigns. Each attack is run for 20 random draws, and the victim
    is a lower-half topic by true yield. `--victim near` instead picks a victim ranked 4–6.
  - **Placement:** fake replies land on the victim's most-replied post (`--place top`) or are spread evenly over its
    posts (`--place spread`).
- **Reported per tier:**
  - clean yield;
  - worst yield over A1–A11 and over A1–A13;
  - the share of attacked days on which the strongest partial attack puts its victim into the top 3.

## Use

```python
import sys; sys.path.insert(0, "bench")
import ccbench as cb

class Responders(cb.Method):
    name = "responders"
    def score(self, views):                      # views: {topic: View} for one (possibly attacked) day
        return {t: len(v.responders) for t, v in views.items()}

S = cb.load("outputs/ccbench_v1")
res = cb.evaluate(cb.reference_methods() + [Responders()], S)
print(cb.summarize(res, S))
```

`python bench/ccbench.py outputs/ccbench_v1` runs the ten reference methods:
- the counts replies, posts, authors and conversational responders;
- three single-side capped counts;
- the cross-anchored count;
- the frozen ranker;
- the gate.

These take about 2 minutes per variant on one CPU. `python bench/test_reference.py` checks that the harness
reproduces `bench/reference_top.json`.

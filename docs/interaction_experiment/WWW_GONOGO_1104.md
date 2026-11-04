# WWW 2027 short paper: go/no-go analyses (2026-11-04)

Script: `scripts/www_gonogo.py` (read-only, CPU, ~7 s) → `outputs/interaction_experiment/www_gonogo/gonogo.json`.
Data: run 475 neighbourhood collection, main population (`llm_v2d`), 2,452 observed seeds.
Deadlines: abstract 2026-11-09, full paper 2026-11-16 (short track, OpenReview, double-blind).

## Verdict

The plan's no-go condition is **not** triggered: after removing feed accounts, ransomware's
interaction rank stays last (11) rather than returning to its post rank. But a risk the plan did not
anticipate appears: the **global** claim ("interaction rate varies by topic beyond what volume
predicts") is not robust to feed exclusion. The paper should lead with the mechanism (feed-made
visibility, self-thread inflation) and report ransomware as directionally consistent, not significant.

## Results

| | all edges | no self-replies/quotes |
|---|---|---|
| CSR, author-clustered 95% CI | 14.5% [12.2, 17.2] | 9.1% [7.6, 10.8] |
| global permutation p, no exclusion | 0.0002 | 0.0003 |
| global p, behavioral_v1 feeds removed (−960 seeds) | 0.044 | 0.026 |
| global p, volume-only feeds removed (−1,079 seeds) | 0.136 | — |
| ransomware rank, posts / interactions (behavioral_v1 removed) | 7 / 11 | 7 / 11 |
| ransomware interaction-per-post vs rest (behavioral_v1 removed) | 0.24 [0.09, 0.64] | 0.25 [0.09, 0.65] |
| ransomware FWER p (behavioral_v1 removed) | 0.33 | 0.66 |

Sanity check: with no exclusion the per-topic FWER reproduces `WWW2027_SHORT_PAPER.md` §3
(ransomware 0.060, data_breach 0.026, vulnerability_cve 0.038).

1. **Self-threads.** 297 of 781 interaction edges (38%) are the seed author's own replies or quotes;
   for ransomware, 12 of 21. `derive.py` counts them. Report both, use the self-excluded version as primary.
2. **Feed-made visibility.** 173 of 249 ransomware seeds come from behavioral_v1 feed accounts
   (212 under the volume-only rule, 159 under actor_v1 `intel_feed_*`).
3. **Ransomware after exclusion.** Still lowest interaction; the pairwise ratio CI excludes 1, but only
   76 seeds remain (37 under volume-only) and it does not survive FWER.
4. **Global claim is fragile.** p rises from 0.0002 to 0.03–0.04 (behavioral_v1) and 0.14 (volume-only).
   behavioral_v1 is defined by zero likes+reposts, which is correlated with the outcome, so the
   outcome-free volume-only rule is the cleaner check — and under it the global test fails.
5. **KEV is not a usable second anchor.** KEV-linked CVE posts: CSR 13.0% [4.9, 25.8], 184 posts / 60
   authors, 69% feed; non-KEV: 7.0% [1.9, 18.2], 115 posts / 20 authors, 91% feed. Intervals overlap.
6. **Prior knowledge of the confound.** Commit `a36b51e` (2026-09-22) already records that registry
   victims appear on Bluesky almost only inside feeds, before the interaction work (`9b6f8cc`, 2026-09-25).
   This shows the feed confound was known in advance, not that low ransomware interaction was a
   pre-specified hypothesis.

## Revised main claim

Ransomware visibility on Bluesky is largely produced by feed accounts, and over a third of measured
"interaction" is authors replying to themselves. Correcting for both changes what the sensor appears to
observe; the remaining topic differences, including ransomware's low interaction, are directionally
consistent but underpowered.

## Still open

Registry 27-event analysis; human annotation (150 double-labelled items, binary H, kappa);
follower-count control; the demo `data.json` vs neighbourhood-count discrepancy.

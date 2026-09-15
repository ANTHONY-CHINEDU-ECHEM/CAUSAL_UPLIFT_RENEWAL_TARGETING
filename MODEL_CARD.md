# Model Card — Causal Uplift Renewal-Discount Targeting

## Intended use
Recommends which renewing accounts should receive a retention discount,
based on estimated causal treatment effect (not just churn propensity).

## Metrics (last run)
See `models/metrics.json` and `models/qini_report.json`.
Typical: predicted-ITE-vs-true-effect correlation ≈ 0.50-0.55 (this repo's
synthetic ground truth allows validating uplift recovery directly, which
real deployments cannot do); Qini coefficient ≈ 0.06-0.10 (positive and
clearly above zero, though below the briefing's aspirational 0.15 target,
which assumes a causal-forest model with confidence-interval-based
targeting — see `docs/EXTENDING.md`); discount-spend reduction ≈ 50-55%
at the chosen targeting threshold; persuadable-segment precision ≈ 0.75-0.80.

## Limitations
The T-learner can suffer from regularization bias when treated/control
base rates differ; a causal forest (EconML) generally reduces this. The
randomized-holdout design assumed here is a real experimental design
requirement for any production deployment — historical confounded data
will NOT give valid uplift estimates.

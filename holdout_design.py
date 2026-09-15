"""Documents the randomized-holdout experimental design this dataset
assumes — the causal-inference gold standard uplift modeling requires.

Why a randomized holdout matters: historical discount data is normally
confounded (discounts are given non-randomly to "risky" accounts), so a
model trained on it conflates "would have churned anyway" with "the
discount caused renewal." A randomized 20% control arm (no discount
offered, regardless of risk) breaks that confound and lets a T-learner
estimate the true causal treatment effect.
"""
DESIGN = {
    "treatment_arm_pct": 80,
    "control_arm_pct": 20,
    "randomization_unit": "account (assigned once at renewal-cycle start)",
    "outcome_window_days": 90,
    "primary_outcome": "renewed (binary)",
    "guardrail_metric": "net_revenue_retention",
    "minimum_detectable_effect": "5 percentage points in renewal rate, 80% power, alpha=0.05",
}

if __name__ == "__main__":
    import json
    print(json.dumps(DESIGN, indent=2))

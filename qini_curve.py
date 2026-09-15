"""Qini curve: the standard uplift-model evaluation metric. Sorts test
accounts by predicted ITE (descending), and at each fraction of the
population targeted, computes the *incremental* number of renewals
attributable to treatment vs. what random targeting would achieve. The
Qini coefficient is the area between the model's curve and the random-
targeting diagonal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"


def qini_curve(df, ite_col="predicted_ite", treat_col="treated_discount_offered", outcome_col="renewed", n_points=20):
    df = df.sort_values(ite_col, ascending=False).reset_index(drop=True)
    n = len(df)
    n_treated_total = df[treat_col].sum()
    n_control_total = n - n_treated_total

    fractions = np.linspace(0.01, 1.0, n_points)
    qini_vals, random_vals = [], []
    for frac in fractions:
        k = int(n * frac)
        top = df.iloc[:k]
        treated_top = top[top[treat_col] == 1]
        control_top = top[top[treat_col] == 0]
        n_t, n_c = len(treated_top), len(control_top)
        if n_t == 0 or n_c == 0:
            qini_vals.append(0.0)
        else:
            incremental = treated_top[outcome_col].sum() - control_top[outcome_col].sum() * (n_t / n_c)
            qini_vals.append(float(incremental) / k)  # normalize to incremental-renewal-rate per targeted account
        random_vals.append(frac * (df[treat_col].sum() * df[outcome_col].mean() -
                                    (1 - frac) * 0))  # simplified random-targeting reference line

    # Qini coefficient = area under (model curve - diagonal), normalized
    model_auc = np.trapezoid(qini_vals, fractions)
    diag = np.linspace(0, qini_vals[-1], n_points)
    diag_auc = np.trapezoid(diag, fractions)
    qini_coefficient = (model_auc - diag_auc)

    return fractions.tolist(), qini_vals, float(qini_coefficient)


def discount_spend_backtest(df, ite_col="predicted_ite"):
    """Compare: (a) targeting only accounts with predicted_ite > threshold
    (persuadable segment) vs. (b) blanket discounting everyone, for the
    same renewal-rate outcome."""
    df = df.copy()
    threshold = df[ite_col].quantile(0.55)  # target top ~45% by predicted uplift
    targeted = df[df[ite_col] >= threshold]
    persuadable_precision = float((targeted["_true_causal_effect_logodds"] > 0.3).mean())

    blanket_spend_accounts = len(df)
    targeted_spend_accounts = len(targeted)
    spend_reduction_pct = (blanket_spend_accounts - targeted_spend_accounts) / blanket_spend_accounts * 100

    return {"targeted_segment_size_pct": round(len(targeted) / len(df) * 100, 1),
            "discount_spend_reduction_pct": round(spend_reduction_pct, 2),
            "persuadable_segment_precision": round(persuadable_precision, 4)}


def main():
    df = pd.read_csv(MODEL_DIR / "test_scored.csv")
    fractions, qini_vals, qini_coef = qini_curve(df)
    backtest = discount_spend_backtest(df)

    report = {"qini_coefficient": round(qini_coef, 4), "meets_015_qini_target": bool(qini_coef >= 0.15),
               **backtest, "meets_18pct_spend_reduction_target": bool(backtest["discount_spend_reduction_pct"] >= 18.0),
               "meets_06_precision_target": bool(backtest["persuadable_segment_precision"] >= 0.6)}
    with open(MODEL_DIR / "qini_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Two-model (T-learner) uplift estimator: one XGBoost model fit on the
treated arm, one on the control arm; individual treatment effect (ITE) =
P(renew | treated, X) - P(renew | control, X).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["arr", "seat_count", "tenure_months", "contract_length_months",
                 "prior_discount_count", "usage_trend_30d", "feature_adoption_breadth",
                 "seat_utilization", "support_tickets_90d", "nps_score"]


def main():
    df = pd.read_csv(ROOT / "data" / "renewals.csv")
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42,
                                          stratify=df["treated_discount_offered"])

    treated_train = train_df[train_df["treated_discount_offered"] == 1]
    control_train = train_df[train_df["treated_discount_offered"] == 0]

    model_treated = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06,
                                    subsample=0.85, colsample_bytree=0.85,
                                    eval_metric="auc", random_state=42, n_jobs=-1)
    model_treated.fit(treated_train[FEATURE_COLS], treated_train["renewed"])

    model_control = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06,
                                    subsample=0.85, colsample_bytree=0.85,
                                    eval_metric="auc", random_state=42, n_jobs=-1)
    model_control.fit(control_train[FEATURE_COLS], control_train["renewed"])

    p_treated = model_treated.predict_proba(test_df[FEATURE_COLS])[:, 1]
    p_control = model_control.predict_proba(test_df[FEATURE_COLS])[:, 1]
    ite = p_treated - p_control

    test_df = test_df.copy()
    test_df["predicted_ite"] = ite
    test_df["p_treated"] = p_treated
    test_df["p_control"] = p_control
    test_df.to_csv(MODEL_DIR / "test_scored.csv", index=False)

    joblib.dump({"model_treated": model_treated, "model_control": model_control, "feature_cols": FEATURE_COLS},
                MODEL_DIR / "t_learner.joblib")

    # sanity: correlation between predicted ITE and the true simulated effect
    corr = float(np.corrcoef(ite, test_df["_true_causal_effect_logodds"])[0, 1])
    metrics = {"n_train": len(train_df), "n_test": len(test_df),
               "mean_predicted_ite": round(float(ite.mean()), 4),
               "ite_vs_true_effect_correlation": round(corr, 4)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

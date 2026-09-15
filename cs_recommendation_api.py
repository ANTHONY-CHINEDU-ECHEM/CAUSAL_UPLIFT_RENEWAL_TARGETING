"""CS-manager discount-recommendation function: given an account's
features, returns the predicted individual treatment effect (ITE) and a
recommendation (offer discount / don't) using the trained T-learner.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models_src"))
from t_learner import FEATURE_COLS

MODEL_DIR = ROOT / "models"


def recommend(account_features: dict, ite_threshold: float = 0.05) -> dict:
    bundle = joblib.load(MODEL_DIR / "t_learner.joblib")
    row = pd.DataFrame([account_features])[FEATURE_COLS]
    p_treated = float(bundle["model_treated"].predict_proba(row)[0, 1])
    p_control = float(bundle["model_control"].predict_proba(row)[0, 1])
    ite = p_treated - p_control
    recommend_discount = ite >= ite_threshold
    return {"predicted_ite": round(ite, 4), "p_renew_with_discount": round(p_treated, 4),
            "p_renew_without_discount": round(p_control, 4), "recommend_discount": recommend_discount}


if __name__ == "__main__":
    example = dict(arr=45000, seat_count=25, tenure_months=22, contract_length_months=12,
                    prior_discount_count=0, usage_trend_30d=-0.3, feature_adoption_breadth=0.4,
                    seat_utilization=0.5, support_tickets_90d=3, nps_score=15)
    print(recommend(example))

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "t_learner.joblib"


def test_randomization_is_balanced():
    df = pd.read_csv(ROOT / "data" / "renewals.csv")
    treat_rate = df["treated_discount_offered"].mean()
    assert 0.75 < treat_rate < 0.85


def test_treated_arm_has_higher_renewal_rate():
    df = pd.read_csv(ROOT / "data" / "renewals.csv")
    treated_rate = df[df.treated_discount_offered == 1]["renewed"].mean()
    control_rate = df[df.treated_discount_offered == 0]["renewed"].mean()
    assert treated_rate > control_rate


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run models_src/t_learner.py first")
def test_ite_recovers_true_effect_direction():
    import json
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert metrics["ite_vs_true_effect_correlation"] > 0.3
    assert metrics["mean_predicted_ite"] > 0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run models_src/t_learner.py first")
def test_recommendation_tool_runs():
    sys.path.insert(0, str(ROOT / "tooling"))
    from cs_recommendation_api import recommend
    example = dict(arr=45000, seat_count=25, tenure_months=22, contract_length_months=12,
                    prior_discount_count=0, usage_trend_30d=-0.3, feature_adoption_breadth=0.4,
                    seat_utilization=0.5, support_tickets_90d=3, nps_score=15)
    result = recommend(example)
    assert "recommend_discount" in result

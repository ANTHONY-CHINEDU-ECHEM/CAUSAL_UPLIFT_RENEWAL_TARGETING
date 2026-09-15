# Causal Uplift Modeling for B2B SaaS Renewal Discount Targeting

Targets retention discounts only at customers who genuinely need one to
renew — using a two-model (T-learner) uplift approach trained on a
randomized discount-offer holdout, evaluated with a Qini curve (the
standard uplift-model metric) rather than plain classification accuracy.

> **Reference-implementation note.** The briefing mentions EconML/CausalML
> causal forests. This repo implements the **T-learner directly with two
> XGBoost models** (`models_src/t_learner.py`) plus a from-scratch **Qini
> curve** (`evaluation/qini_curve.py`) — the core uplift-modeling technique
> without the extra dependency weight. A causal-forest upgrade path
> (EconML) is documented in `docs/EXTENDING.md`.

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_data.py            # synthetic randomized-holdout renewal data -> data/
python models_src/t_learner.py          # trains treatment/control models -> models/
python evaluation/qini_curve.py         # Qini coefficient, discount-spend backtest
```

## Structure
| Path | Purpose |
|---|---|
| `experiment_design/holdout_design.py` | Documents the randomized-holdout experimental design |
| `data/generate_data.py` | Synthesizes 60k renewal events with a true 20% randomized holdout |
| `models_src/t_learner.py` | Two-model uplift estimator (treated vs. control XGBoost) |
| `evaluation/qini_curve.py` | Qini coefficient + discount-spend-reduction backtest |
| `tooling/cs_recommendation_api.py` | CS-manager discount-recommendation function |

## License
MIT — portfolio/demonstration use.

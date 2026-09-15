# Extending this reference implementation

1. **Causal forest.** Replace `models_src/t_learner.py`'s two-model
   approach with EconML's `CausalForestDML` for confidence-interval-aware
   heterogeneous treatment-effect estimation — should improve the Qini
   coefficient toward the briefing's 0.15 target.
2. **Real randomized holdout.** This repo's data is simulated WITH a true
   randomization; a real deployment must run an actual randomized
   discount-offer experiment (see `experiment_design/holdout_design.py`)
   before any uplift model can be trusted.
3. **Multi-treatment uplift.** Extend to multiple discount tiers /
   added-value offers instead of a single binary treatment.
4. **CS tool integration.** Wire `tooling/cs_recommendation_api.py` behind
   a Gainsight/Totango webhook for the renewal workflow.

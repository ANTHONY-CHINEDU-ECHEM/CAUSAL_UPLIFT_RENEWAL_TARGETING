**CAUSAL UPLIFT RENEWAL TARGETING: HETEROGENEOUS TREATMENT EFFECT (HTE) MODEL FOR SAAS RETENTION**

**PROJECT OVERVIEW AND BUSINESS RATIONALE**

Subscription businesses frequently allocate retention budgets inefficiently through blanket discounting strategies or standard churn propensity models. Blanket discounting applies financial incentives across entire customer segments, causing unnecessary margin erosion on accounts that would renew organically or accounts that remain unrecoverable. Conventional churn propensity models estimate the baseline probability of cancellation without accounting for customer responsiveness to financial intervention.

This repository implements a heterogeneous treatment effect pipeline designed to calculate Individual Treatment Effects (ITE). By isolating customers whose renewal decisions are directly caused by a discount offer, the model optimises capital allocation, protects recurring revenue, and prevents unnecessary financial concessions.

<img width="1306" height="816" alt="UR DASHBOARD" src="https://github.com/user-attachments/assets/a19e4dc9-f5f3-47ac-bd8c-cc49cfde4be3" />


**MATHEMATICAL FORMALISATION AND CAUSAL FRAMEWORK**


The methodology evaluates the causal impact of a treatment variable W on an outcome variable Y conditional on a feature vector X.

Treatment Variable (W): Binary indicator where W = 1 denotes receipt of a retention discount offer and W = 0 denotes assignment to the control group.

Outcome Variable (Y): Binary indicator where Y = 1 denotes successful subscription renewal and Y = 0 denotes customer churn.

Individual Treatment Effect (ITE): Defined as the difference between potential outcomes under treatment and control conditions:


ITE(x) = E[Y(1) - Y(0)| at X=x] = P(Y=1| X=x, W=1)-P(Y=1| X=x, W=0)

The objective is to target accounts where the ITE exceeds a positive economic threshold, ensuring that discounts are restricted to persuadable segments.

**DATA ARCHITECTURE AND EXPERIMENTAL DESIGN**

The model is trained and validated on a synthetic cohort of 30,000 enterprise renewal accounts partitioned into a 21,000 instance training set and a 9,000 instance test set.


**EXPERIMENTAL ASSIGNMENT:**

To satisfy the unconfoundedness assumption required for causal inference, treatment assignment was randomised. Exactly 24,079 accounts (80.3%) received the discount offer, while 5,921 accounts (19.7%) served as the randomised control group.

**FEATURE SCHEMA:**

Ten quantitative attributes categorised into three distinct operational groups:

**1. Firmographics:** Annual recurring revenue (arr), seat allocation volume (seat_count), and contractual commitment duration (contract_length_months).

**2. Relationship History:** Account lifetime duration (tenure_months), historical discount frequency (prior_discount_count), support case volume over ninety days (support_tickets_90d), and net promoter score (nps_score).

**3. Product Engagement:** Thirty-day usage trajectory (usage_trend_30d), feature adoption breadth (feature_adoption_breadth), and license utilisation rate (seat_utilization).

**EMPIRICAL OBSERVATION**

Feature importance analysis indicates that recent product usage velocity (usage_trend_30d) carries approximately three times the predictive weight of static firmographic metrics when estimating treatment responsiveness.


**MODELLING METHODOLOGY**

The modelling implementation utilises a Two-Model Learner (T-Learner) framework to estimate heterogeneous effects by training separate estimators for each treatment arm:

- Treated Estimator (model_treated): An XGBoost binary classifier trained exclusively on the subset of data where
W = 1.

- Control Estimator (model_control): An identical XGBoost binary classifier trained exclusively on the subset of data where W = 0.

- Inference Execution: For any target account $x$, the individual treatment effect is derived by subtracting the predicted probability of the control model from the predicted probability of the treated model:

<img width="344" height="52" alt="Screenshot 2026-09-15 at 14 44 37" src="https://github.com/user-attachments/assets/0c9aa6d2-4ce7-4e64-8aae-fa2aefebf1d9" />

This separation prevents structural regularisation bias that occurs when treatment indicators are merely appended as categorical features in a single estimator.


**EVALUATION METRICS AND EMPIRICAL RESULTS**

Model performance is validated using specialised causal evaluation metrics on the held-out test cohort of 9,000 accounts.

**Simulation Ground-Truth Alignment:** The predicted individual treatment effects achieve a Pearson correlation coefficient of 0.522 against the synthetic ground-truth log-odds effect.

**Ranking Performance:** The evaluation yields a Qini coefficient of 0.080, confirming positive ranking performance superior to random assignment.

**Targeting Policy Backtest:** Restricting discount deployment to the top 45% of accounts ordered by predicted ITE (applying a classification threshold at the 55th percentile, equivalent to an ITE of 0.110) produces the following operational results:

- A 55% reduction in total financial discount expenditure compared to blanket distribution models.

- A 77.5% precision rate when identifying genuinely persuadable accounts characterised by a true simulated effect exceeding 0.3 log-odds.

- An 8.3 percentage point net increase in overall test-set renewal rates.


**REPOSITORY STRUCTURE**

CAUSAL_UPLIFT_RENEWAL_TARGETING/

│

├── data/                  # Cohort generation scripts, training data, and test splits

├── notebooks/             # Exploratory analysis and causal validation workflows

├── src/                   # Core pipeline modules for data ingestion and T-Learner execution

├── models/                # Serialized XGBoost model binaries

├── README.md              # Detailed technical project documentation

└── requirements.txt       # Pinned Python package dependencies


**INSTALLATION AND EXECUTION INSTRUCTIONS**

1. CLONE THE REPOSITORY

   git clone https://github.com/ANTHONY-CHINEDU-ECHEM/CAUSAL_UPLIFT_RENEWAL_TARGETING.git
cd CAUSAL_UPLIFT_RENEWAL_TARGETING

2. ESTABLISH A VIRTUAL ENVIRONMENT AND INSTALLATION DEPENDENCIES

   python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. EXECUTE THE PIPELINE

   Run the primary orchestration scripts located within the src/ directory to train the estimators and generate evaluation reports.

**OPERATIONAL CONSTRAINTS AND FUTURE ROADMAP**

**Methodological Limitation:** Observational data lacking randomised assignment must not be substituted for experimental data, as unmeasured confounding variables invalidate the unconfoundedness assumption.

**Roadmap Integration**: Future iterations will replace the T-Learner architecture with EconML Causal Forest Double Machine Learning (DML) estimators to generate asymptotic confidence intervals and improve treatment effect granularity across sparse feature spaces.

**LICENSE**

This project is distributed under the terms of the MIT License. Review the LICENSE file for additional details.

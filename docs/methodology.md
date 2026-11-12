# Implementation Methodology: Predicting Insurance Claim Approval Using Apache Spark

**Course**: INT315 – Cluster Computing  
**Student Name**: Rishabh Kumar  
**Roll Number**: 49 | **Registration Number**: 12310491 | **Section**: K2P23PD  
**Project Title**: Predicting Insurance Claim Approval Using Apache Spark  

---

## 1. System Architecture Overview

The distributed analytics and machine learning pipeline is designed as an end-to-end, multi-stage architecture utilizing **Apache Spark 4.3.0 / PySpark** running in `local[*]` multi-core execution mode. The pipeline enforces strict separation between data generation, distributed ingestion, data cleaning, feature engineering, distributed machine learning model training, model evaluation, and inference.

```
+-------------------------------------------------------------------------------+
|                       SYNTHETIC DATA GENERATOR (Seed=42)                      |
|                  120,000 Insurance Claim Records (CSV RFC 4180)               |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      APACHE SPARK DISTRIBUTED INGESTION                       |
|           Explicit StructType Schema Ingestion into Spark DataFrame          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       DISTRIBUTED DATA CLEANING & HYGIENE                     |
|           Deduplication, Domain Validation, Imputation of Missing Values      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    FEATURE ENGINEERING & CATALYST TRANSFORMATIONS             |
|       claim_to_premium_ratio, claim_to_estimate_ratio, Tenure & Delay Bins    |
+---------------------------------------+---------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
+---------------------------------------+ +-------------------------------------+
|        DISTRIBUTED EDA (SPARK SQL)    | |        SPARK MLLIB PIPELINES        |
|  - Claim Vol & Approval Distribution  | |  - Train/Test Split (80:20 Split)   |
|  - Policy Type & Severity Analysis    | |  - StringIndexer -> OneHotEncoder   |
|  - Documentation & Payment Analysis   | |  - VectorAssembler -> ML Pipeline   |
|  - Export Aggregated CSVs & Charts    | |  - Logistic Regression (Baseline)   |
+---------------------------------------+ |  - Random Forest Classifier (Tree)  |
                                          +------------------+------------------+
                                                             |
                                                             v
                                          +-------------------------------------+
                                          |     RIGOROUS EVALUATION & SELECTION |
                                          |  - Accuracy, F1, Precision, Recall  |
                                          |  - Minority Class Recall (Rejected) |
                                          |  - ROC-AUC, PR-AUC, Confusion Matrix|
                                          |  - Model Persistence to Disk        |
                                          +------------------+------------------+
                                                             |
                                                             v
+------------------------------------------------------------+------------------+
|                    STREAMLIT INTERACTIVE ANALYTICS & INFERENCE                |
|      Executive KPI Cards, Risk Visualizations, Real-Time Claim Predictor      |
+-------------------------------------------------------------------------------+
```

---

## 2. Synthetic Data Formulation & Probability Model

To create a credible machine learning task without data leakage or trivial rules, claim approval ($Y \in \{0, 1\}$, where $1 = \text{Approved}$, $0 = \text{Rejected}$) was modeled via a logistic probability function:

$$P(Y = 1 \mid \mathbf{X}) = \frac{1}{1 + e^{-z(\mathbf{X})}}$$

The latent log-odds score $z(\mathbf{X})$ is defined as:

$$z = \beta_0 + \sum_{j=1}^{p} w_j f_j(\mathbf{X}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

Where:
- $\beta_0 = 1.05$ (calibrated baseline for ~70% approval rate)
- $w_{\text{policy\_active}} = -3.8$ if inactive ($+0.4$ if active)
- $w_{\text{documentation}} \in \{+1.3 \text{ (Complete)}, -0.5 \text{ (Partial)}, -2.6 \text{ (Missing)}\}$
- $w_{\text{payment\_status}} \in \{+0.8 \text{ (Paid)}, -0.3 \text{ (Pending)}, -2.0 \text{ (Overdue)}\}$
- $w_{\text{submission\_delay}} = +0.4$ if $\le 7$ days, $-0.7$ if $\le 30$ days, $-1.8$ if $> 30$ days
- $w_{\text{estimate\_ratio}} = -1.1$ if $\frac{\text{claim\_amount}}{\text{estimate}} > 1.25$
- $w_{\text{prior\_claims}} = +0.35$ if $0$ claims, $-0.8$ if $\ge 3$ claims
- $w_{\text{reported}} = +0.5$ if reported, $-0.9$ if unreported
- $w_{\text{tenure}} = +0.45$ if $> 48$ months, $-0.35$ if $< 6$ months
- $\epsilon \sim \mathcal{N}(0, 0.75)$ (continuous stochastic variation)

A claim is adjudicated as **Approved** if $U(0, 1) < P(Y = 1 \mid \mathbf{X})$, and **Rejected** otherwise.

---

## 3. Spark Data Loading and Preprocessing

1. **Explicit Schema Parsing**:  
   Ingesting data using an explicit `StructType` prevents costly schema-inference passes across all 120,000 rows.
2. **Deduplication**:  
   Spark DataFrame transformation `dropDuplicates(["claim_id"])` removes redundant submissions.
3. **Imputation**:  
   Missing numeric entries in non-critical fields (`customer_age`, `previous_claims`, `repair_or_treatment_estimate`) are imputed using domain medians/defaults.
4. **Boundary Validation**:  
   Negative claim amounts, invalid policy types, and corrupt timestamps are filtered out using Spark Boolean expressions.

---

## 4. Feature Engineering

1. **Claim-to-Premium Ratio**:
   $$\text{CPR} = \frac{\text{claim\_amount}}{\text{annual\_premium}}$$
   Reflects the financial claim magnitude relative to the recurring policy revenue.
2. **Claim-to-Estimate Ratio**:
   $$\text{CER} = \frac{\text{claim\_amount}}{\max(\text{estimate}, 1.0)}$$
   Detects excessive or inflated claim submissions exceeding authorized assessments.
3. **Binned Categorical Cohorts**:
   - `policy_tenure_category`: `<12m (New)`, `12-48m (Established)`, `>48m (Mature)`
   - `submission_delay_category`: `<=3d (Immediate)`, `4-14d (Normal)`, `>14d (Delayed)`
   - `previous_claim_category`: `0 (None)`, `1-2 (Few)`, `3+ (Frequent)`

---

## 5. Machine Learning Pipelines (Spark MLlib)

The machine learning workflow is encapsulated in a unified `Pipeline`:

1. **`StringIndexer`**: Maps string categorical labels into numerical indices.
2. **`OneHotEncoder`**: Converts category indices into binary vector representations.
3. **`VectorAssembler`**: Gathers all continuous numerical features and one-hot vectors into a single feature vector `features`.
4. **Estimators**:
   - **Baseline Model**: `LogisticRegression(maxIter=25, regParam=0.01)`
   - **Ensemble Model**: `RandomForestClassifier(numTrees=40, maxDepth=8, seed=42)`

---

## 6. Model Evaluation and Selection Criteria

Models are evaluated on the 20% holdout test partition using:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Weighted Precision & Recall**
- **Weighted F1-Score**
- **ROC-AUC (Area Under ROC Curve)**
- **PR-AUC (Area Under Precision-Recall Curve)**
- **Recall for Rejected Class (Minority Class)**:
  $$\text{Recall}_{\text{Rejected}} = \frac{TN}{TN + FP}$$
  In insurance claim risk management, failing to identify an ineligible claim (False Positive) incurs direct financial loss. Hence, recall on the minority Rejected class is heavily weighted in the model selection criteria.

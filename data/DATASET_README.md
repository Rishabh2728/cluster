# Dataset Specification: Synthetic Insurance Claims Approval Dataset

**Course**: INT315 – Cluster Computing  
**Student Name**: Rishabh Kumar  
**Roll Number**: 49 | **Registration Number**: 12310491 | **Section**: K2P23PD  
**Project**: Predicting Insurance Claim Approval Using Apache Spark  

---

## 1. Dataset Status & Ethical Declaration

> **IMPORTANT ACADEMIC DECLARATION**:  
> This dataset is **entirely synthetically generated** for academic research and educational evaluation in course INT315 (Cluster Computing).  
> It **does not represent real customer, policyholder, or proprietary insurance company data**. All claim numbers, policy IDs, loss details, and claim decisions are generated via Python algorithms (`src/dataset_generator.py`) using fixed pseudo-random routines. No real-world automated claim decisions should be based on this dataset or the trained models.

---

## 2. Generation Methodology & Determinism

- **Generator Module**: `src/dataset_generator.py`
- **Pseudo-Random Number Generator Seed**: `42` (`numpy.random` and `random`)
- **Total Record Count**: 120,000 insurance claim lines
- **Target Distribution**: Approximately 70% Approved, 30% Rejected (moderate class imbalance)
- **Output File Path**: `data/raw/insurance_claims.csv`
- **Output File Format**: Comma Separated Values (RFC 4180, UTF-8 encoded)

---

## 3. Schema Architecture

| Feature Name | Spark SQL Type | Logical Range / Domain | Description | Predictive Role |
|:---|:---|:---|:---|:---|
| `claim_id` | `StringType` | `CLM-0000001` ... | Unique claim transaction key | Non-predictive identifier (excluded from ML) |
| `policy_id` | `StringType` | `POL-100000` ... | Policy contract key | Non-predictive identifier (excluded from ML) |
| `policy_type` | `StringType` | Health, Motor, Life, Property | Category of insurance coverage | Categorical Feature |
| `claim_amount` | `DoubleType` | $500.00 – $150,000.00 | Monetary indemnity requested | Continuous Numerical Feature |
| `annual_premium` | `DoubleType` | $300.00 – $6,000.00 | Annual policy premium charged | Continuous Numerical Feature |
| `policy_tenure_months` | `IntegerType` | 1 – 240 months | Duration customer held policy | Continuous Numerical Feature |
| `customer_age` | `IntegerType` | 18 – 82 years | Age of insured claimant | Continuous Numerical Feature |
| `incident_type` | `StringType` | Collision, Theft, Emergency, etc. | Classification of loss incident | Categorical Feature |
| `incident_severity` | `StringType` | Low, Medium, High | Assessed loss impact tier | Categorical Feature |
| `previous_claims` | `IntegerType` | 0 – 6 claims | Prior historical claims submitted | Discrete Numerical Feature |
| `documentation_status` | `StringType` | Complete, Partial, Missing | Verification level of proof | Categorical Feature |
| `policy_active` | `StringType` | Yes, No | Status of policy at loss date | Categorical Feature |
| `premium_payment_status` | `StringType` | Paid, Pending, Overdue | Account status of premiums | Categorical Feature |
| `claim_submission_delay` | `IntegerType` | 0 – 90 days | Elapsed days before reporting | Discrete Numerical Feature |
| `incident_reported` | `StringType` | Yes, No | Filed with official authorities | Categorical Feature |
| `repair_or_treatment_estimate` | `DoubleType` | $500.00 – $160,000.00 | Professional loss appraisal | Continuous Numerical Feature |
| `claim_approval` | `StringType` | `Approved`, `Rejected` | Final adjudication decision | **Supervised ML Target Label** |

---

## 4. Feature Correlations & Adjudication Logic

The synthetic generation process simulates realistic underwriting decision dynamics:
1. **Policy Status (`policy_active`)**: Inactive policies experience a high rejection rate due to lack of legal coverage at loss time.
2. **Documentation (`documentation_status`)**: Claims with `Complete` documentation exhibit higher approval odds compared to `Partial` or `Missing`.
3. **Account Standing (`premium_payment_status`)**: `Overdue` accounts incur heightened rejection probabilities due to policy lapses.
4. **Appraisal Discrepancies**: The ratio `claim_amount / repair_or_treatment_estimate` flags excessive padding or inflated loss claims.
5. **Submission Latency (`claim_submission_delay`)**: Reporting delays exceeding 30–60 days trigger closer investigation and higher denial rates.
6. **Prior Claim Frequency (`previous_claims`)**: Repeated historical claims moderate the baseline approval probability.
7. **Stochastic Realism**: Gaussian noise ($\sigma = 0.75$) prevents deterministic rules, mimicking authentic human underwriter variance.

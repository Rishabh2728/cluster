"""
Synthetic Insurance Claims Dataset Generator
INT315 - Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD

Generates a realistic, reproducible synthetic dataset of insurance claim records
using deterministic random seed 42. Features realistic probabilistic correlations
with the target variable (claim_approval: Approved / Rejected).
Target approval distribution: ~68-72% Approved, ~28-32% Rejected.
"""

import csv
import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config import DATASET_CONFIG, ensure_directories

POLICY_TYPES = ["Health", "Motor", "Life", "Property"]
POLICY_TYPE_PROBS = [0.35, 0.35, 0.15, 0.15]

INCIDENT_TYPES = {
    "Health": ["Medical Emergency", "Surgical Procedure", "Hospitalization", "Outpatient Treatment"],
    "Motor": ["Vehicle Collision", "Theft/Burglary", "Windshield Damage", "Vandalism"],
    "Life": ["Accidental Death", "Critical Illness", "Disability", "Terminal Illness"],
    "Property": ["Water Damage", "Fire Damage", "Theft/Burglary", "Natural Hazard"]
}

INCIDENT_SEVERITY = ["Low", "Medium", "High"]
SEVERITY_PROBS = [0.45, 0.38, 0.17]

DOCUMENTATION_STATUS = ["Complete", "Partial", "Missing"]
DOC_PROBS = [0.65, 0.25, 0.10]

POLICY_ACTIVE = ["Yes", "No"]
POLICY_ACTIVE_PROBS = [0.93, 0.07]

PREMIUM_PAYMENT_STATUS = ["Paid", "Pending", "Overdue"]
PAYMENT_STATUS_PROBS = [0.78, 0.14, 0.08]

INCIDENT_REPORTED = ["Yes", "No"]
REPORTED_PROBS = [0.85, 0.15]


def sigmoid(x):
    """Numerically stable sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def generate_synthetic_claims(
    target_records: int = DATASET_CONFIG["target_records"],
    seed: int = DATASET_CONFIG["random_seed"],
    output_path: Path = DATASET_CONFIG["raw_csv_path"]
) -> dict:
    """
    Generates synthetic insurance claim transactions deterministically in high-speed batches.
    
    Returns:
        dict: Generation summary metrics and statistics.
    """
    ensure_directories()
    np.random.seed(seed)
    random.seed(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset Generator] Generating {target_records:,} insurance claim records with seed {seed}...")

    header = [
        "claim_id",
        "policy_id",
        "policy_type",
        "claim_amount",
        "annual_premium",
        "policy_tenure_months",
        "customer_age",
        "incident_type",
        "incident_severity",
        "previous_claims",
        "documentation_status",
        "policy_active",
        "premium_payment_status",
        "claim_submission_delay",
        "incident_reported",
        "repair_or_treatment_estimate",
        "claim_approval"
    ]

    approved_count = 0
    rejected_count = 0
    missing_count = 0
    records_written = 0

    batch_size = 10000
    rows_batch = []

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i in range(1, target_records + 1):
            claim_id = f"CLM-{i:07d}"
            policy_id = f"POL-{(i % 65000) + 100000:06d}"

            # 1. Policy Type
            p_type_idx = np.random.choice(len(POLICY_TYPES), p=POLICY_TYPE_PROBS)
            policy_type = POLICY_TYPES[p_type_idx]

            # 2. Incident Type depending on policy type
            incidents_for_type = INCIDENT_TYPES[policy_type]
            incident_type = np.random.choice(incidents_for_type)

            # 3. Incident Severity
            severity_idx = np.random.choice(len(INCIDENT_SEVERITY), p=SEVERITY_PROBS)
            incident_severity = INCIDENT_SEVERITY[severity_idx]

            # 4. Customer Age (18 to 82)
            customer_age = int(np.random.normal(loc=44, scale=14))
            customer_age = max(18, min(82, customer_age))

            # 5. Policy Tenure (months: 1 to 240)
            tenure_months = int(np.random.exponential(scale=36)) + 1
            tenure_months = min(240, tenure_months)

            # 6. Previous claims (0 to 6)
            prev_claims = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.55, 0.22, 0.12, 0.06, 0.03, 0.015, 0.005]))

            # 7. Documentation status
            doc_idx = np.random.choice(len(DOCUMENTATION_STATUS), p=DOC_PROBS)
            doc_status = DOCUMENTATION_STATUS[doc_idx]

            # 8. Policy active
            policy_active = np.random.choice(POLICY_ACTIVE, p=POLICY_ACTIVE_PROBS)

            # 9. Premium payment status
            pay_idx = np.random.choice(len(PREMIUM_PAYMENT_STATUS), p=PAYMENT_STATUS_PROBS)
            premium_status = PREMIUM_PAYMENT_STATUS[pay_idx]

            # 10. Claim submission delay in days (0 to 90)
            delay_days = int(np.random.exponential(scale=12))
            delay_days = min(90, delay_days)

            # 11. Incident reported
            reported = np.random.choice(INCIDENT_REPORTED, p=REPORTED_PROBS)

            # 12. Financials based on policy type and severity
            if policy_type == "Health":
                base_est = np.random.uniform(1500, 35000)
                annual_premium = round(np.random.uniform(600, 3800), 2)
            elif policy_type == "Motor":
                base_est = np.random.uniform(800, 22000)
                annual_premium = round(np.random.uniform(500, 2500), 2)
            elif policy_type == "Life":
                base_est = np.random.uniform(10000, 85000)
                annual_premium = round(np.random.uniform(800, 5000), 2)
            else:  # Property
                base_est = np.random.uniform(2000, 50000)
                annual_premium = round(np.random.uniform(700, 4200), 2)

            if incident_severity == "High":
                base_est *= np.random.uniform(1.3, 1.8)
            elif incident_severity == "Low":
                base_est *= np.random.uniform(0.5, 0.8)

            repair_or_treatment_estimate = round(base_est, 2)

            # Claim amount with realistic deviation from estimate
            claim_deviation = np.random.uniform(0.85, 1.35)
            claim_amount = round(repair_or_treatment_estimate * claim_deviation, 2)

            # =========================================================================
            # Realistic Probabilistic Scoring Model for Claim Approval
            # Calibrated baseline for ~70% approved and 30% rejected
            # =========================================================================
            z = 0.40  # Calibrated baseline log-odds

            # Policy active penalty (strongest criterion)
            if policy_active == "No":
                z -= 3.8
            else:
                z += 0.2

            # Documentation status
            if doc_status == "Complete":
                z += 1.0
            elif doc_status == "Partial":
                z -= 0.6
            elif doc_status == "Missing":
                z -= 2.6

            # Premium payment status
            if premium_status == "Paid":
                z += 0.6
            elif premium_status == "Pending":
                z -= 0.4
            elif premium_status == "Overdue":
                z -= 2.0

            # Submission delay
            if delay_days <= 7:
                z += 0.3
            elif delay_days > 30:
                z -= 0.8
            if delay_days > 60:
                z -= 1.2

            # Claim vs Estimate ratio (inflated claims are suspicious)
            ratio = claim_amount / max(repair_or_treatment_estimate, 1.0)
            if ratio > 1.25:
                z -= 1.1
            elif ratio <= 1.02:
                z += 0.2

            # Previous claims
            if prev_claims == 0:
                z += 0.25
            elif prev_claims >= 3:
                z -= 0.8
            if prev_claims >= 5:
                z -= 1.2

            # Incident reported to authorities
            if reported == "Yes":
                z += 0.4
            else:
                z -= 0.9

            # Policy tenure
            if tenure_months > 48:
                z += 0.35
            elif tenure_months < 6:
                z -= 0.35

            # Continuous Gaussian noise
            z += np.random.normal(loc=0.0, scale=0.8)

            approval_prob = sigmoid(z)
            decision = "Approved" if np.random.random() < approval_prob else "Rejected"

            if decision == "Approved":
                approved_count += 1
            else:
                rejected_count += 1

            # Optional missing value injection (~2% in customer_age, previous_claims, estimate)
            age_str = str(customer_age)
            prev_str = str(prev_claims)
            est_str = f"{repair_or_treatment_estimate:.2f}"

            if np.random.random() < DATASET_CONFIG["missing_rate"]:
                choice = np.random.choice(["age", "prev", "estimate"])
                if choice == "age":
                    age_str = ""
                elif choice == "prev":
                    prev_str = ""
                else:
                    est_str = ""
                missing_count += 1

            rows_batch.append([
                claim_id,
                policy_id,
                policy_type,
                f"{claim_amount:.2f}",
                f"{annual_premium:.2f}",
                tenure_months,
                age_str,
                incident_type,
                incident_severity,
                prev_str,
                doc_status,
                policy_active,
                premium_status,
                delay_days,
                reported,
                est_str,
                decision
            ])
            records_written += 1

            if len(rows_batch) >= batch_size:
                writer.writerows(rows_batch)
                rows_batch.clear()

        if rows_batch:
            writer.writerows(rows_batch)
            rows_batch.clear()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    approval_pct = round((approved_count / records_written) * 100, 2)
    rejection_pct = round((rejected_count / records_written) * 100, 2)

    stats = {
        "total_records": records_written,
        "seed": seed,
        "raw_csv_path": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
        "approved_records": approved_count,
        "approved_pct": approval_pct,
        "rejected_records": rejected_count,
        "rejected_pct": rejection_pct,
        "injected_missing_records": missing_count,
        "missing_pct": round((missing_count / records_written) * 100, 2)
    }

    print(f"[Dataset Generator] Synthetic generation completed: {records_written:,} records ({file_size_mb:.2f} MB)")
    print(f"  - Approved Claims: {approved_count:,} ({approval_pct}%)")
    print(f"  - Rejected Claims: {rejected_count:,} ({rejection_pct}%)")
    print(f"  - Missing Injections: {missing_count:,} ({stats['missing_pct']}%)")

    return stats


if __name__ == "__main__":
    generate_synthetic_claims()

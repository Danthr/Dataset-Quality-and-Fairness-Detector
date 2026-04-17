"""
Fairness Auditor
Core engine for dataset fairness auditing
"""

import re
import pandas as pd
from typing import Dict, List, Optional
import logging
from .metrics import FairnessMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditor:

    PROTECTED_ATTRIBUTE_KEYWORDS = {
        "gender": ["gender", "sex"],
        "age": ["age", "age_group"],
        "race": ["race", "ethnicity"],
        "religion": ["religion"],
        "caste": ["caste"],
        "disability": ["disability"],
        "marital": ["marital"],
        "nationality": ["nationality"],
    }

    def __init__(self):
        self.metrics = FairnessMetrics()

    # -----------------------------
    # DETECT PROTECTED ATTRIBUTES
    # -----------------------------
    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        detected = []

        for col in df.columns:
            col_lower = col.lower()

            for keyword_group in self.PROTECTED_ATTRIBUTE_KEYWORDS.values():
                if any(
                    re.search(rf"\b{re.escape(keyword)}\b", col_lower)
                    for keyword in keyword_group
                ):
                    detected.append(col)
                    break

        return detected

    # -----------------------------
    # ELIGIBILITY CHECK
    # -----------------------------
    def evaluate_audit_eligibility(
        self,
        df: pd.DataFrame,
        user_protected_attrs: Optional[List[str]] = None,
    ) -> Dict:

        if user_protected_attrs:
            valid = [a for a in user_protected_attrs if a in df.columns]

            return {
                "audit_allowed": len(valid) > 0,
                "detected_attributes": valid,
                "source": "manual_override",
                "message": (
                    "Manual protected attributes accepted"
                    if valid else "No valid attributes provided"
                ),
            }

        auto = self.detect_protected_attributes(df)

        return {
            "audit_allowed": len(auto) > 0,
            "detected_attributes": auto,
            "source": "auto_detected",
            "message": (
                "Protected attributes detected"
                if auto else "No protected attributes found"
            ),
        }

    # -----------------------------
    # SELECT OUTCOME COLUMN
    # -----------------------------
    def select_outcome(self, df, protected_attrs):

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        non_protected_numeric = [
            col for col in numeric_cols if col not in protected_attrs
        ]

        if non_protected_numeric:
            return non_protected_numeric[-1]

        if numeric_cols:
            return numeric_cols[-1]

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        non_protected_cat = [
            col for col in categorical_cols if col not in protected_attrs
        ]

        if not non_protected_cat:
            return None

        return max(non_protected_cat, key=lambda c: df[c].nunique())

    # -----------------------------
    # SINGLE ATTRIBUTE AUDIT
    # -----------------------------
    def audit_single(self, df, protected_attr, outcome_attr):

        df = df.copy()

        # -----------------------------
        # NUMERIC BINNING
        # -----------------------------
        if pd.api.types.is_numeric_dtype(df[protected_attr]):
            try:
                df[protected_attr] = pd.qcut(
                    df[protected_attr],
                    q=2,
                    duplicates="drop"
                )
            except Exception:
                return {
                    "protected_attribute": protected_attr,
                    "error": "Failed to bin numeric attribute"
                }

        # -----------------------------
        # OUTCOME HANDLING (FIXED)
        # -----------------------------
        if pd.api.types.is_numeric_dtype(df[outcome_attr]):
            unique_vals = df[outcome_attr].dropna().unique()

            # Binary case
            if set(unique_vals).issubset({0, 1}):
                outcome_col = outcome_attr
                favorable = 1
            else:
                threshold = df[outcome_attr].median()
                df["_fav"] = df[outcome_attr] > threshold
                outcome_col = "_fav"
                favorable = True
        else:
            favorable = df[outcome_attr].mode()[0]
            outcome_col = outcome_attr

        # -----------------------------
        # GROUP DETECTION
        # -----------------------------
        groups = df[protected_attr].dropna().unique()

        if len(groups) < 2:
            return {
                "protected_attribute": protected_attr,
                "error": "Need at least 2 groups"
            }

        # -----------------------------
        # RATE CALCULATION
        # -----------------------------
        rates = {}

        for g in groups:
            mask = df[protected_attr] == g
            total = mask.sum()

            positive = (
                df.loc[mask, outcome_col] == favorable
            ).sum()

            rates[g] = positive / total if total > 0 else 0

        privileged = max(rates, key=rates.get)
        unprivileged = min(rates, key=rates.get)

        # -----------------------------
        # EDGE CASE: SAME RATES
        # -----------------------------
        if privileged == unprivileged:
            return {
                "protected_attribute": protected_attr,
                "outcome_attribute": outcome_attr,
                "message": "All groups have equal outcome rates",
                "disparate_impact": 1.0,
                "demographic_parity": 0.0,
                "spd": 0.0,
                "is_fair": True,
                "verdict": "FAIR",
            }

        # -----------------------------
        # METRICS
        # -----------------------------
        di = self.metrics.disparate_impact(
            df,
            protected_attr,
            outcome_col,
            favorable,
            privileged,
            unprivileged
        )

        dp = self.metrics.demographic_parity(
            df,
            protected_attr,
            outcome_col,
            favorable
        )

        spd = self.metrics.statistical_parity_difference(
            df,
            protected_attr,
            outcome_col,
            favorable,
            privileged
        )

        # -----------------------------
        # SPD CHECK
        # -----------------------------
        spd_value = spd.get("statistical_parity_difference")

        if spd_value is None:
            spd_is_fair = False
        else:
            spd_is_fair = abs(spd_value) <= 0.1

        # -----------------------------
        # FINAL FAIRNESS
        # -----------------------------
        is_fair = (
            di["is_fair"]
            and dp["is_fair"]
            and spd_is_fair
        )

        return {
            "protected_attribute": protected_attr,
            "outcome_attribute": outcome_attr,
            "privileged_group": str(privileged),
            "unprivileged_group": str(unprivileged),
            "disparate_impact": di["disparate_impact_score"],
            "demographic_parity": dp["parity_difference"],
            "spd": spd_value,
            "is_fair": is_fair,
            "verdict": "FAIR" if is_fair else "UNFAIR",
        }

    # -----------------------------
    # FULL AUDIT
    # -----------------------------
    def audit_all(self, df, protected_attrs=None, outcome_attr=None):

        eligibility = self.evaluate_audit_eligibility(df, protected_attrs)

        if not eligibility["audit_allowed"]:
            return {
                "error": "No protected attributes found",
                "audit_allowed": False,
                "detected_attributes": [],
            }

        protected_attrs = eligibility["detected_attributes"]

        if outcome_attr is None:
            outcome_attr = self.select_outcome(df, protected_attrs)

        if outcome_attr is None or outcome_attr not in df.columns:
            return {
                "error": "No valid outcome column found",
                "audit_allowed": False,
            }

        results = {}

        for attr in protected_attrs:
            results[attr] = self.audit_single(
                df,
                attr,
                outcome_attr
            )

        return {
            "audit_allowed": True,
            "detected_attributes": protected_attrs,
            "outcome_attribute": outcome_attr,
            "results": results,
        }
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
    """Audits datasets for fairness and bias"""

    PROTECTED_ATTRIBUTE_KEYWORDS = {
        "gender": ["gender", "sex", "male", "female"],
        "age": ["age", "age_group", "dob", "birth_year"],
        "race": ["race", "ethnicity", "ethnic_group"],
        "religion": ["religion", "faith"],
        "caste": ["caste", "category", "reservation"],
        "income": ["income", "salary_band", "economic_class"],
        "disability": ["disability", "special_needs"],
        "marital": ["marital", "marital_status"],
        "nationality": ["nationality", "citizenship"],
    }

    def __init__(self):
        self.metrics = FairnessMetrics()

    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Detect protected attributes using expanded keyword mapping.
        Uses word-boundary regex matching.
        """
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

        logger.info(f"Detected protected attributes: {detected}")
        return detected

    def evaluate_audit_eligibility(
        self,
        df: pd.DataFrame,
        user_protected_attrs: Optional[List[str]] = None
    ) -> Dict:
        """
        Decide whether fairness audit should proceed.
        Supports both manual override and auto-detection.
        """
        if user_protected_attrs:
            valid_attrs = [
                attr for attr in user_protected_attrs
                if attr in df.columns
            ]

            return {
                "audit_allowed": len(valid_attrs) > 0,
                "detected_attributes": valid_attrs,
                "source": "manual_override",
                "message": (
                    "Manual protected attributes accepted"
                    if valid_attrs
                    else "No valid manual protected attributes found"
                )
            }

        auto_detected = self.detect_protected_attributes(df)

        return {
            "audit_allowed": len(auto_detected) > 0,
            "detected_attributes": auto_detected,
            "source": "auto_detected" if auto_detected else "none",
            "message": (
                "Protected attributes detected"
                if auto_detected
                else "No protected attributes detected"
            )
        }

    def audit_single(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome=None,
    ) -> Dict:
        logger.info(
            f"Running fairness audit for '{protected_attr}' against '{outcome_attr}'"
        )

        df_copy = df.copy()

        if favorable_outcome is None:
            if pd.api.types.is_numeric_dtype(df_copy[outcome_attr]):
                threshold = df_copy[outcome_attr].median()
                df_copy["_favorable"] = df_copy[outcome_attr] > threshold
                outcome_column = "_favorable"
                favorable_outcome = True
            else:
                favorable_outcome = df_copy[outcome_attr].mode()[0]
                outcome_column = outcome_attr
        else:
            outcome_column = outcome_attr

        groups = df_copy[protected_attr].dropna().unique()

        if len(groups) < 2:
            return {
                "protected_attribute": protected_attr,
                "error": "At least 2 groups required for fairness audit",
            }

        privileged_group = groups[0]
        unprivileged_group = groups[1]

        di_result = self.metrics.disparate_impact(
            df_copy,
            protected_attr,
            outcome_column,
            favorable_outcome,
            privileged_group,
            unprivileged_group,
        )

        dp_result = self.metrics.demographic_parity(
            df_copy,
            protected_attr,
            outcome_column,
            favorable_outcome,
        )

        spd_result = self.metrics.statistical_parity_difference(
            df_copy,
            protected_attr,
            outcome_column,
            favorable_outcome,
            privileged_group,
        )

        is_fair = di_result["is_fair"] or dp_result["is_fair"]

        return {
            "protected_attribute": protected_attr,
            "outcome_attribute": outcome_attr,
            "disparate_impact": di_result.get("disparate_impact_score"),
            "demographic_parity": dp_result.get("parity_difference"),
            "spd": spd_result.get("statistical_parity_difference"),
            "is_fair": is_fair,
            "verdict": "FAIR" if is_fair else "UNFAIR — Bias Detected",
        }

    def audit_all(
        self,
        df: pd.DataFrame,
        protected_attrs: Optional[List[str]] = None,
        outcome_attr: Optional[str] = None,
    ) -> Dict:
        """
        Run fairness audit only if audit is eligible.
        """
        eligibility = self.evaluate_audit_eligibility(
            df,
            protected_attrs
        )

        if not eligibility["audit_allowed"]:
            return {
                "error": "Fairness audit requires at least one protected attribute",
                "audit_allowed": False,
                "detected_attributes": [],
            }

        protected_attrs = eligibility["detected_attributes"]

        if outcome_attr is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            non_protected_numeric = [
                col for col in numeric_cols
                if col not in protected_attrs
            ]

            if non_protected_numeric:
                outcome_attr = non_protected_numeric[-1]
            elif numeric_cols:
                outcome_attr = numeric_cols[-1]
            else:
                categorical_cols = df.select_dtypes(
                    include=["object"]
                ).columns.tolist()

                non_protected_cat = [
                    col for col in categorical_cols
                    if col not in protected_attrs
                ]

                if not non_protected_cat:
                    return {
                        "error": "No suitable outcome column found in dataset"
                    }

                outcome_attr = max(
                    non_protected_cat,
                    key=lambda c: df[c].nunique()
                )

        if outcome_attr not in df.columns:
            return {
                "error": (
                    f"Outcome column '{outcome_attr}' not found. "
                    f"Available: {df.columns.tolist()}"
                )
            }

        results = {
            "audit_allowed": True,
            "detected_attributes": protected_attrs,
            "outcome_attribute": outcome_attr,
            "results": {}
        }

        for attr in protected_attrs:
            results["results"][attr] = self.audit_single(
                df,
                attr,
                outcome_attr
            )

        return results

    def generate_report(self, results: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("FAIRNESS AUDIT REPORT")
        lines.append("=" * 60)

        audit_results = results.get("results", {})

        for attr, result in audit_results.items():
            lines.append(f"\nProtected Attribute: {attr}")

            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            else:
                lines.append(f"  Verdict:            {result.get('verdict')}")
                lines.append(
                    f"  Disparate Impact:   {result.get('disparate_impact')}"
                )
                lines.append(
                    f"  Demographic Parity: {result.get('demographic_parity')}"
                )
                lines.append(f"  SPD:                {result.get('spd')}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
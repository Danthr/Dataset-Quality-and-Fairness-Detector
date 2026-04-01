"""
Fairness Auditor
Core engine for dataset fairness auditing
"""

import pandas as pd
from typing import Dict, List, Optional
import logging
from .metrics import FairnessMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditor:
    """Audits datasets for fairness and bias"""

    def __init__(self):
        self.metrics = FairnessMetrics()

    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Detect protected attributes based on column names
        """
        protected_keywords = [
            "gender",
            "sex",
            "race",
            "ethnicity",
            "age",
            "religion",
            "disability",
            "marital",
            "nationality",
        ]

        detected = []

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                detected.append(col)

        logger.info(f"Detected protected attributes: {detected}")
        return detected

    def audit_single(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome: Optional[any] = None,
    ) -> Dict:
        """
        Audit fairness for one protected attribute
        """
        logger.info(f"Running fairness audit for {protected_attr}")

        df_copy = df.copy()

        # Auto-detect favorable outcome
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

        result = {
            "protected_attribute": protected_attr,
            "disparate_impact": di_result.get("disparate_impact_score"),
            "demographic_parity": dp_result.get("parity_difference"),
            "spd": spd_result.get("statistical_parity_difference"),
            "is_fair": is_fair,
            "verdict": "FAIR" if is_fair else "UNFAIR — Bias Detected",
        }

        return result

    def audit_all(
        self,
        df: pd.DataFrame,
        protected_attrs: Optional[List[str]] = None,
        outcome_attr: Optional[str] = None,
    ) -> Dict:
        """
        Audit all detected or provided protected attributes
        """
        if protected_attrs is None:
            protected_attrs = self.detect_protected_attributes(df)

        if not protected_attrs:
            return {
                "error": "No protected attributes detected"
            }

        if outcome_attr is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if not numeric_cols:
                return {
                    "error": "No numeric outcome column found"
                }

            outcome_attr = numeric_cols[0]

        results = {}

        for attr in protected_attrs:
            results[attr] = self.audit_single(
                df,
                attr,
                outcome_attr,
            )

        return results

    def generate_report(self, results: Dict) -> str:
        """
        Generate simple readable report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("FAIRNESS AUDIT REPORT")
        lines.append("=" * 60)

        for attr, result in results.items():
            lines.append(f"\nProtected Attribute: {attr}")
            lines.append(f"Verdict: {result.get('verdict')}")
            lines.append(
                f"Disparate Impact: {result.get('disparate_impact')}"
            )
            lines.append(
                f"Demographic Parity: {result.get('demographic_parity')}"
            )
            lines.append(f"SPD: {result.get('spd')}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
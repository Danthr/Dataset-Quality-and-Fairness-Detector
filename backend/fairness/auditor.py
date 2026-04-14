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

    def __init__(self):
        self.metrics = FairnessMetrics()

    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Detect protected attributes based on column names.
        Uses whole-word matching so columns like 'message_id', 'teenage_group',
        or 'percentage' are NOT falsely flagged just because they contain a
        protected keyword as a substring.
        Case-insensitive.
        """
        protected_keywords = [
            "gender", "sex", "race", "ethnicity", "age",
            "religion", "disability", "marital", "nationality",
        ]

        # FIX 2c: Use word-boundary regex instead of plain substring matching.
        # Plain 'in' check was causing false positives:
        #   'message_id'   contains 'age'  → wrongly flagged
        #   'teenage_group' contains 'age' → wrongly flagged
        #   'percentage'   contains 'age'  → wrongly flagged
        # \b matches a word boundary, so only actual standalone words match.
        detected = []
        for col in df.columns:
            col_lower = col.lower()
            if any(re.search(rf'\b{keyword}\b', col_lower) for keyword in protected_keywords):
                detected.append(col)

        logger.info(f"Detected protected attributes: {detected}")
        return detected

    def audit_single(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome=None,
    ) -> Dict:
        """
        Audit fairness for one protected attribute against one outcome column.

        If the outcome column is numeric, uses median as threshold —
        above median = positive outcome.
        If categorical, uses the most common value as the favorable outcome.
        """
        logger.info(f"Running fairness audit for '{protected_attr}' against '{outcome_attr}'")

        df_copy = df.copy()

        # --- Determine favorable outcome and which column to compare against ---
        if favorable_outcome is None:
            if pd.api.types.is_numeric_dtype(df_copy[outcome_attr]):
                # Numeric outcome: above median is "positive"
                threshold = df_copy[outcome_attr].median()
                df_copy["_favorable"] = df_copy[outcome_attr] > threshold
                outcome_column = "_favorable"
                favorable_outcome = True
            else:
                # Categorical outcome: most common value is "positive"
                favorable_outcome = df_copy[outcome_attr].mode()[0]
                outcome_column = outcome_attr
        else:
            outcome_column = outcome_attr

        # Need at least 2 groups to compare
        groups = df_copy[protected_attr].dropna().unique()
        if len(groups) < 2:
            return {
                "protected_attribute": protected_attr,
                "error": "At least 2 groups required for fairness audit",
            }

        # First group found is treated as privileged, second as unprivileged.
        # For a proper audit the user can specify these explicitly.
        privileged_group = groups[0]
        unprivileged_group = groups[1]

        di_result = self.metrics.disparate_impact(
            df_copy, protected_attr, outcome_column,
            favorable_outcome, privileged_group, unprivileged_group,
        )

        dp_result = self.metrics.demographic_parity(
            df_copy, protected_attr, outcome_column, favorable_outcome,
        )

        spd_result = self.metrics.statistical_parity_difference(
            df_copy, protected_attr, outcome_column,
            favorable_outcome, privileged_group,
        )

        # FAIR if at least one of the two primary metrics passes
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
        Audit all detected or provided protected attributes.

        If protected_attrs is None, auto-detects from column names.
        If outcome_attr is None, auto-selects the best column:
          1. Prefers a non-protected numeric column
          2. Falls back to any numeric column
          3. FIX 3: Falls back to best categorical column if no numeric exists
             (most unique values, excluding protected columns themselves)
        """
        if protected_attrs is None:
            protected_attrs = self.detect_protected_attributes(df)

        if not protected_attrs:
            return {"error": "No protected attributes detected"}

        if outcome_attr is None:
            protected_keywords = [
                'gender', 'sex', 'race', 'ethnicity', 'age',
                'religion', 'disability', 'marital', 'nationality'
            ]

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            # Filter out columns that are also protected attributes
            non_protected_numeric = [
                col for col in numeric_cols
                if not any(re.search(rf'\b{kw}\b', col.lower()) for kw in protected_keywords)
            ]

            if non_protected_numeric:
                # Best case: non-protected numeric column
                outcome_attr = non_protected_numeric[-1]
            elif numeric_cols:
                # Fallback: any numeric column
                outcome_attr = numeric_cols[-1]
            else:
                # FIX 3: No numeric columns at all — use best categorical column.
                # "Best" = most unique values (most discriminating), excluding
                # the protected attributes themselves so we don't audit gender vs gender.
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                non_protected_cat = [
                    col for col in categorical_cols
                    if not any(re.search(rf'\b{kw}\b', col.lower()) for kw in protected_keywords)
                ]

                if not non_protected_cat:
                    return {"error": "No suitable outcome column found in dataset"}

                # Pick the categorical column with the most unique values
                outcome_attr = max(non_protected_cat, key=lambda c: df[c].nunique())

            logger.info(f"Auto-selected outcome column: '{outcome_attr}'")

        # Confirm outcome column exists
        if outcome_attr not in df.columns:
            return {
                "error": f"Outcome column '{outcome_attr}' not found. "
                         f"Available: {df.columns.tolist()}"
            }

        results = {}
        for attr in protected_attrs:
            if attr not in df.columns:
                results[attr] = {"error": f"Column '{attr}' not found in dataset"}
                continue
            results[attr] = self.audit_single(df, attr, outcome_attr)

        return results

    def generate_report(self, results: Dict) -> str:
        """
        Generate a simple readable text report from audit results.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("FAIRNESS AUDIT REPORT")
        lines.append("=" * 60)

        for attr, result in results.items():
            lines.append(f"\nProtected Attribute: {attr}")
            if "error" in result:
                lines.append(f"  Error: {result['error']}")
            else:
                lines.append(f"  Verdict:            {result.get('verdict')}")
                lines.append(f"  Disparate Impact:   {result.get('disparate_impact')}")
                lines.append(f"  Demographic Parity: {result.get('demographic_parity')}")
                lines.append(f"  SPD:                {result.get('spd')}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
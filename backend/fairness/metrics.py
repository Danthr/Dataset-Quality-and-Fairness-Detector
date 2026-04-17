"""
Fairness Metrics
Implements fairness metric calculations using pandas and numpy
"""

import pandas as pd
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessMetrics:

    def disparate_impact(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome,
        privileged_group,
        unprivileged_group,
    ) -> Dict:

        privileged_mask = df[protected_attr] == privileged_group
        unprivileged_mask = df[protected_attr] == unprivileged_group

        privileged_total = privileged_mask.sum()
        unprivileged_total = unprivileged_mask.sum()

        privileged_positive = (
            df.loc[privileged_mask, outcome_attr] == favorable_outcome
        ).sum()

        unprivileged_positive = (
            df.loc[unprivileged_mask, outcome_attr] == favorable_outcome
        ).sum()

        privileged_rate = (
            privileged_positive / privileged_total
            if privileged_total > 0 else 0.0
        )

        unprivileged_rate = (
            unprivileged_positive / unprivileged_total
            if unprivileged_total > 0 else 0.0
        )

        if privileged_rate == 0:
            return {
                "disparate_impact_score": None,
                "privileged_rate": round(privileged_rate, 4),
                "unprivileged_rate": round(unprivileged_rate, 4),
                "is_fair": False,
                "interpretation": "Undetermined — privileged rate is 0",
            }

        di_score = unprivileged_rate / privileged_rate

        return {
            "disparate_impact_score": round(di_score, 4),
            "privileged_rate": round(privileged_rate, 4),
            "unprivileged_rate": round(unprivileged_rate, 4),
            "is_fair": di_score >= 0.8,
        }

    def demographic_parity(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome,
    ) -> Dict:

        groups = df[protected_attr].dropna().unique()

        rates = []

        for group in groups:
            mask = df[protected_attr] == group
            total = mask.sum()

            positive = (
                df.loc[mask, outcome_attr] == favorable_outcome
            ).sum()

            rate = positive / total if total > 0 else 0
            rates.append(rate)

        parity_difference = max(rates) - min(rates) if rates else 0

        return {
            "parity_difference": round(parity_difference, 4),
            "is_fair": parity_difference < 0.1,
        }

    def statistical_parity_difference(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        outcome_attr: str,
        favorable_outcome,
        reference_group,
    ) -> Dict:

        ref_mask = df[protected_attr] == reference_group
        ref_total = ref_mask.sum()

        ref_positive = (
            df.loc[ref_mask, outcome_attr] == favorable_outcome
        ).sum()

        ref_rate = ref_positive / ref_total if ref_total > 0 else 0

        groups = df[protected_attr].dropna().unique()

        spd_values = {}

        for group in groups:
            if group == reference_group:
                continue

            mask = df[protected_attr] == group
            total = mask.sum()

            positive = (
                df.loc[mask, outcome_attr] == favorable_outcome
            ).sum()

            rate = positive / total if total > 0 else 0

            spd_values[str(group)] = round(rate - ref_rate, 4)

        if spd_values:
            overall_spd = max(spd_values.values(), key=abs)
        else:
            overall_spd = None

        return {
            "statistical_parity_difference": overall_spd,
            "all_group_spd": spd_values,
        }
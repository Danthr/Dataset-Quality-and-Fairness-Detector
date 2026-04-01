"""
AI Explainer Module
Generates plain-English explanations for quality and fairness reports
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AIExplainer:
    """Generates human-readable backend explanations"""

    def explain_quality(self, quality_result: Dict) -> Dict:
        """
        Generate explanation for data quality report
        """
        overall_grade = quality_result.get("overall_grade", "N/A")
        overall_score = quality_result.get("overall_score", 0)

        if overall_grade == "A":
            summary = (
                "The dataset quality is excellent. "
                "The data is complete, valid, consistent, "
                "and safe for further fairness analysis."
            )

        elif overall_grade == "B":
            summary = (
                "The dataset quality is good with minor issues. "
                "Proceed with fairness analysis, but review anomalies."
            )

        elif overall_grade == "C":
            summary = (
                "The dataset quality is acceptable, "
                "but data review is recommended."
            )

        else:
            summary = (
                "The dataset quality is poor. "
                "Data cleaning is strongly recommended."
            )

        return {
            "type": "quality_explanation",
            "grade": overall_grade,
            "score": overall_score,
            "summary": summary,
        }

    def explain_fairness(self, fairness_result: Dict) -> Dict:
        """
        Generate explanation for fairness audit
        """
        explanations = {}

        for attr, result in fairness_result.items():
            verdict = result.get("verdict", "UNKNOWN")

            if "UNFAIR" in verdict:
                explanation = (
                    f"The protected attribute '{attr}' "
                    f"shows signs of potential bias. "
                    f"This indicates unequal treatment "
                    f"across demographic groups."
                )

            else:
                explanation = (
                    f"The protected attribute '{attr}' "
                    f"passes fairness checks and "
                    f"shows balanced outcomes."
                )

            explanations[attr] = {
                "verdict": verdict,
                "explanation": explanation,
            }

        return {
            "type": "fairness_explanation",
            "details": explanations,
        }

    def generate_full_report(
        self,
        quality_result: Dict,
        fairness_result: Dict,
    ) -> Dict:
        """
        Combined explanation report
        """
        quality_explanation = self.explain_quality(
            quality_result
        )

        fairness_explanation = self.explain_fairness(
            fairness_result
        )

        return {
            "report_type": "full_ai_explanation",
            "quality_summary": quality_explanation,
            "fairness_summary": fairness_explanation,
            "note": (
                "This is a placeholder rule-based "
                "AI explanation engine. "
                "Claude API integration will replace this later."
            ),
        }
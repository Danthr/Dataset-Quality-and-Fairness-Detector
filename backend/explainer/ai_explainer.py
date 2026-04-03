"""
AI Explanation Engine
Uses Anthropic Claude API to generate plain English
explanations of data quality and fairness audit results
"""

import os
import anthropic
from typing import Dict
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class AIExplainer:
    """
    Generates plain English explanations using Claude API
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. "
                "AI explanations will not work."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 1000

    def _call_claude(self, prompt: str) -> str:
        """
        Send prompt to Claude API and return response text
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    def explain_quality(self, quality_result: Dict) -> Dict:
        """
        Generate plain English explanation for data quality report
        """
        try:
            grade = quality_result.get("overall_grade", "N/A")
            score = quality_result.get("overall_score", 0)
            completeness = quality_result.get(
                "completeness", {}
            ).get("score", 0)
            validity = quality_result.get(
                "validity", {}
            ).get("score", 0)
            consistency = quality_result.get(
                "consistency", {}
            ).get("score", 0)
            uniqueness = quality_result.get(
                "uniqueness", {}
            ).get("score", 0)
            missing_pct = quality_result.get(
                "completeness", {}
            ).get("missing_pct", 0)
            n_violations = quality_result.get(
                "consistency", {}
            ).get("n_violations", 0)
            duplicate_rows = quality_result.get(
                "uniqueness", {}
            ).get("duplicate_rows", 0)
            total_invalid = quality_result.get(
                "validity", {}
            ).get("total_invalid", 0)

            prompt = f"""You are a data quality expert explaining results to a non-technical audience.

Here are the data quality scores for an uploaded dataset:
- Overall Grade: {grade}
- Overall Score: {score:.2%}
- Completeness Score: {completeness:.2%} ({missing_pct}% missing values)
- Validity Score: {validity:.2%} ({total_invalid} invalid values detected)
- Consistency Score: {consistency:.2%} ({n_violations} cross-column contradictions)
- Uniqueness Score: {uniqueness:.2%} ({duplicate_rows} duplicate rows)

Write a data quality explanation with exactly these three sections:

Section 1 — Plain Explanation:
2-3 sentences explaining what these scores mean. No technical jargon. Written for a manager who has never seen data quality scores before.

Section 2 — Key Concerns:
1-2 sentences describing the most important issue in this dataset that needs attention, if any.

Section 3 — Action Items:
Exactly 3 bullet points of specific, actionable steps the user should take before running a fairness audit on this data.

Keep the entire response under 250 words."""

            explanation_text = self._call_claude(prompt)

            logger.info("Quality explanation generated successfully")

            return {
                "type": "quality_explanation",
                "grade": grade,
                "score": score,
                "explanation": explanation_text,
            }

        except Exception as e:
            logger.error(f"Error generating quality explanation: {e}")
            return {
                "type": "quality_explanation",
                "grade": quality_result.get("overall_grade", "N/A"),
                "score": quality_result.get("overall_score", 0),
                "explanation": (
                    "AI explanation unavailable. "
                    "Please check your ANTHROPIC_API_KEY."
                ),
                "error": str(e),
            }

    def explain_fairness(self, fairness_result: Dict) -> Dict:
        """
        Generate plain English explanation for fairness audit results
        """
        try:
            # Build a summary of findings for the prompt
            findings = []

            for attr, result in fairness_result.items():
                if "error" in result:
                    continue

                verdict = result.get("verdict", "UNKNOWN")
                di = result.get("disparate_impact", "N/A")
                dp = result.get("demographic_parity", "N/A")
                spd = result.get("spd", "N/A")

                findings.append(
                    f"- Attribute: {attr} | "
                    f"Verdict: {verdict} | "
                    f"Disparate Impact: {di} (threshold >= 0.8) | "
                    f"Demographic Parity Diff: {dp} (threshold < 0.1) | "
                    f"Statistical Parity Diff: {spd}"
                )

            if not findings:
                return {
                    "type": "fairness_explanation",
                    "explanation": "No fairness results available to explain.",
                }

            findings_text = "\n".join(findings)

            prompt = f"""You are a fairness and ethics expert explaining algorithmic bias results to a non-technical audience.

Here are the fairness audit results for an uploaded dataset:

{findings_text}

Metric thresholds for reference:
- Disparate Impact: must be >= 0.8 to pass (the 4/5ths rule). Lower means more bias.
- Demographic Parity Difference: must be < 0.1 to pass. Higher means more bias.
- Statistical Parity Difference: values close to 0 are fair. Negative means the group is disadvantaged.

Write a fairness explanation with exactly these three sections:

Section 1 — Plain Explanation:
2-3 sentences explaining what these bias findings mean in plain language. No technical jargon. Written for an HR manager or compliance officer.

Section 2 — Real World Impact:
1-2 sentences with a concrete example of what could go wrong if a machine learning model is trained on this data as-is.

Section 3 — Action Items:
Exactly 3 bullet points of specific, actionable steps to address the bias found in this dataset.

Keep the entire response under 300 words."""

            explanation_text = self._call_claude(prompt)

            logger.info("Fairness explanation generated successfully")

            return {
                "type": "fairness_explanation",
                "explanation": explanation_text,
            }

        except Exception as e:
            logger.error(f"Error generating fairness explanation: {e}")
            return {
                "type": "fairness_explanation",
                "explanation": (
                    "AI explanation unavailable. "
                    "Please check your ANTHROPIC_API_KEY."
                ),
                "error": str(e),
            }

    def generate_full_report(
        self,
        quality_result: Dict,
        fairness_result: Dict,
    ) -> Dict:
        """
        Generate combined quality + fairness explanation report
        """
        logger.info("Generating full AI explanation report")

        quality_explanation = self.explain_quality(quality_result)
        fairness_explanation = self.explain_fairness(fairness_result)

        return {
            "report_type": "full_ai_explanation",
            "quality_summary": quality_explanation,
            "fairness_summary": fairness_explanation,
        }
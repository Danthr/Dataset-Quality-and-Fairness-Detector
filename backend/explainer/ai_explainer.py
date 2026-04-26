"""
AI Explanation Engine (FINAL STABLE VERSION)
- Uses google.genai (new SDK)
- Uses gemini-2.5-flash (stable free-tier model)
- Includes retry handling for rate limits
"""

import os
import time
import logging
from typing import Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()
logger = logging.getLogger(__name__)


class AIExplainer:

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            logger.warning("GOOGLE_API_KEY not set.")

        self.client = genai.Client(api_key=api_key)

        # ✅ FINAL MODEL (stable + free-tier safe)
        self.model = "gemini-2.5-flash"

    def _call_gemini(self, prompt: str) -> str:
        """
        Gemini call with retry logic (handles 429 errors)
        """
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                print("\n=== GEMINI RAW RESPONSE ===")
                print(response)

                if response and response.text:
                    return response.text.strip()

                print("⚠️ Empty Gemini response")
                return "Explanation unavailable."

            except Exception as e:
                error_msg = str(e)

                # Retry on rate limit
                if "429" in error_msg:
                    print(f"⚠️ Rate limit hit, retrying... ({attempt+1}/3)")
                    time.sleep(2)
                    continue

                print("\n❌ GEMINI ERROR:", error_msg)
                logger.error(f"Gemini API error: {error_msg}")
                return "Explanation unavailable."

        return "Explanation unavailable."

    def explain_quality(self, quality_result: Dict) -> Dict:
        try:
            prompt = f"""
You are a data quality expert.

Explain this result in simple terms:

{quality_result}

Give:
1. Summary (2-3 sentences)
2. Key issues
3. 3 actionable improvements
"""

            explanation = self._call_gemini(prompt)

            return {
                "type": "quality_explanation",
                "grade": quality_result.get("overall_grade"),
                "score": quality_result.get("overall_score"),
                "explanation": explanation,
            }

        except Exception as e:
            return {
                "type": "quality_explanation",
                "explanation": "Explanation unavailable.",
                "error": str(e),
            }

    def explain_fairness(self, fairness_result: Dict) -> Dict:
        try:
            results = fairness_result.get("results", {})

            findings = []
            for attr, result in results.items():
                if not isinstance(result, dict):
                    continue

                findings.append(
                    f"{attr}: {result.get('verdict')} | "
                    f"DI={result.get('disparate_impact')} | "
                    f"SPD={result.get('spd')}"
                )

            if not findings:
                return {
                    "type": "fairness_explanation",
                    "explanation": "No fairness results available."
                }

            prompt = f"""
You are a fairness expert.

Explain these results in simple terms:

{chr(10).join(findings)}

Give:
1. Summary (2-3 sentences)
2. Real-world impact
3. 3 actionable recommendations
"""

            explanation = self._call_gemini(prompt)

            return {
                "type": "fairness_explanation",
                "explanation": explanation,
            }

        except Exception as e:
            return {
                "type": "fairness_explanation",
                "explanation": "Explanation unavailable.",
                "error": str(e),
            }

    def generate_full_report(self, quality_result: Dict, fairness_result: Dict) -> Dict:
        logger.info("Generating full AI explanation report")

        return {
            "report_type": "full_ai_explanation",
            "quality_summary": self.explain_quality(quality_result),
            "fairness_summary": self.explain_fairness(fairness_result),
        }
"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List, Optional
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        self.enabled = config.get("enabled", True)
        self.toxicity_threshold = config.get("toxicity_threshold", 0.5)

        # Optional Guardrails AI integration. Similar to InputGuardrail this is
        # best-effort and fails open if dependency is missing.
        self.guard = None
        if self.enabled:
            try:
                from guardrails import Guard  # type: ignore
                from guardrails.validators import (  # type: ignore
                    ToxicLanguage,
                    PIIFilter,
                )

                self.guard = Guard().use_many(
                    ToxicLanguage(threshold=self.toxicity_threshold),
                    PIIFilter(),
                )
            except Exception:
                self.guard = None

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        """
        if not self.enabled:
            return {
                "valid": True,
                "violations": [],
                "sanitized_output": response,
            }

        violations: List[Dict[str, Any]] = []
        sanitized_output = response

        # Prefer Guardrails AI if available
        if self.guard is not None:
            try:
                result = self.guard.validate(response)
                passed = getattr(result, "validation_passed", None)
                errors = getattr(result, "errors", None)
                output = getattr(result, "output", response)

                if isinstance(passed, bool) and errors is not None:
                    for err in errors:
                        if isinstance(err, dict):
                            violations.append(err)
                        else:
                            violations.append(
                                {
                                    "validator": getattr(
                                        err, "validator_name", "external_guard"
                                    ),
                                    "reason": str(
                                        getattr(err, "message", repr(err))
                                    ),
                                    "severity": getattr(
                                        err, "severity", "medium"
                                    ),
                                }
                            )

                    return {
                        "valid": passed and not violations,
                        "violations": violations,
                        "sanitized_output": output,
                    }
            except Exception:
                pass

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_bias(response))
        violations.extend(self._check_factual_consistency(response, sources or []))

        if violations:
            sanitized_output = self._sanitize(response, violations)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": sanitized_output,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        """
        violations: List[Dict[str, Any]] = []
        matched_spans: Dict[str, List[str]] = {}

        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]*)?\(?\d{3}\)?[-.\s]*\d{3}[-.\s]*\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            # Avoid over-triggering on DOI/URLs; require common credit card prefixes and separators
            "credit_card": r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                matched_spans[pii_type] = matches

        for pii_type, matches in matched_spans.items():
            violations.append(
                {
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches,
                }
            )

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.
        """

        violations: List[Dict[str, Any]] = []
        lowered = text.lower()

        categories = [
            {
                "name": "self_harm",
                "phrases": [
                    "kill myself",
                    "kill yourself",
                    "commit suicide",
                    "self harm",
                    "hurt myself",
                ],
                "severity": "high",
            },
            {
                "name": "violence",
                "phrases": [
                    "shoot up",
                    "stab",
                    "bomb",
                    "terrorist attack",
                    "explosive device",
                ],
                "severity": "high",
            },
            {
                "name": "illegal_activity",
                "phrases": [
                    "buy drugs",
                    "sell drugs",
                    "launder money",
                    "hack into",
                    "bypass security",
                ],
                "severity": "medium",
            },
        ]

        def _phrase_matches(text: str, phrase: str) -> bool:
            """Use word-boundary regex to avoid substring false positives."""
            # Replace spaces with flexible whitespace and add word boundaries
            pattern = r"\\b" + re.escape(phrase).replace("\\ ", r"\\s+") + r"\\b"
            return re.search(pattern, text) is not None

        for category in categories:
            matches = [
                phrase
                for phrase in category["phrases"]
                if _phrase_matches(lowered, phrase)
            ]
            if matches:
                violations.append(
                    {
                        "validator": "harmful_content",
                        "category": category["name"],
                        "reason": f"Detected {category['name'].replace('_', ' ')} content",
                        "severity": category["severity"],
                        "matches": matches,
                    }
                )

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        """
        violations: List[Dict[str, Any]] = []
        lowered = response.lower()

        red_flag_phrases = [
            "i made this up",
            "i fabricated this",
            "i cannot verify this",
            "this might be inaccurate",
            "no evidence",
            "not fact checked",
            "citation needed",
        ]

        for phrase in red_flag_phrases:
            if phrase in lowered:
                violations.append(
                    {
                        "validator": "factual_consistency",
                        "reason": f"Potential hallucination or uncertainty: '{phrase}'",
                        "severity": "medium",
                        "matches": [phrase],
                    }
                )

        # Basic citation check: references like [1] without provided sources
        citation_markers = re.findall(r"\[\d+\]", response)
        if citation_markers and not sources:
            violations.append(
                {
                    "validator": "factual_consistency",
                    "reason": "Contains citation markers but no sources provided",
                    "severity": "low",
                    "matches": citation_markers,
                }
            )

        # If sources exist but response never references them, flag for review
        if sources and not citation_markers:
            violations.append(
                {
                    "validator": "factual_consistency",
                    "reason": "Sources attached but no citations found in response",
                    "severity": "low",
                }
            )

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:

        violations: List[Dict[str, Any]] = []
        lowered = text.lower()

        biased_terms = [
            ("ageism", ["senile", "geezer", "old folks", "over the hill", "past their prime", "slow learner (age-related context)", "not tech-savvy (as a stereotype)", "elderly burden",]),
            (
                "gender",
                ["hysterical women", "man up", "like a girl", "boys club"],
            ),
            (
                "race",
                ["illegal alien", "third world", "colored people"],
            ),
            (
               "disability", [ "crippled", "handicapped", "wheelchair-bound", "suffers from", "invalid", ],
            ),
        ]

        for category, phrases in biased_terms:
            matches = [phrase for phrase in phrases if phrase in lowered]
            if matches:
                violations.append(
                    {
                        "validator": "bias",
                        "category": category,
                        "reason": f"Potential biased language referencing {category}",
                        "severity": "medium",
                        "matches": matches,
                    }
                )

        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        """
        sanitized = text

        for violation in violations:
            validator = violation.get("validator")
            matches: Optional[List[str]] = violation.get("matches")

            if not matches:
                continue

            replacement = "[REDACTED]"
            if validator == "harmful_content":
                replacement = "[CONTENT WARNING]"
            elif validator == "bias":
                replacement = "[BIAS FLAG]"

            for match in matches:
                sanitized = sanitized.replace(match, replacement)

        return sanitized

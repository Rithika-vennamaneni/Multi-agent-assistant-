"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Basic configuration for local checks
        self.enabled = config.get("enabled", True)
        self.min_length = config.get("min_length", 1)
        self.max_length = config.get("max_length", 4096)
        self.toxicity_threshold = config.get("toxicity_threshold", 0.5)

        # Best-effort integration with Guardrails AI if available.
        # This keeps initialization lightweight and fails open if
        # the library or validators are not installed.
        self.guard = None
        if self.enabled:
            try:
                from guardrails import Guard  # type: ignore
                from guardrails.validators import (  # type: ignore
                    ValidLength,
                    ToxicLanguage,
                )

                self.guard = Guard().use_many(
                    ValidLength(min=self.min_length, max=self.max_length),
                    ToxicLanguage(threshold=self.toxicity_threshold),
                )
            except Exception:
                # Fall back to the lightweight, built-in checks in `validate`.
                self.guard = None

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        """
        # Fast path: guardrail disabled
        if not self.enabled:
            return {
                "valid": True,
                "violations": [],
                "sanitized_input": query,
            }

        normalized_query = (query or "").strip()
        sanitized_input = " ".join(normalized_query.split())
        violations: List[Dict[str, Any]] = []

        if not sanitized_input:
            return {
                "valid": False,
                "violations": [
                    {
                        "validator": "length",
                        "reason": "Query is empty after sanitization",
                        "severity": "low",
                    }
                ],
                "sanitized_input": "",
            }

        # Prefer external Guardrails-style validators if available.
        if self.guard is not None:
            try:
                result = self.guard.validate(sanitized_input)
                passed = getattr(result, "validation_passed", None)
                errors = getattr(result, "errors", None)
                output = getattr(result, "output", sanitized_input)

                if isinstance(passed, bool) and errors is not None:
                    for err in errors:
                        # Best-effort normalization of error objects
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
                        "sanitized_input": output,
                    }
            except Exception:
                # Fall back to lightweight local checks below.
                pass

        # Local length checks using configured bounds
        if len(sanitized_input) < self.min_length:
            violations.append(
                {
                    "validator": "length",
                    "reason": "Query too short",
                    "severity": "low",
                }
            )

        if len(sanitized_input) > self.max_length:
            violations.append(
                {
                    "validator": "length",
                    "reason": "Query too long",
                    "severity": "medium",
                }
            )
            sanitized_input = sanitized_input[: self.max_length]

        # Additional heuristic checks
        violations.extend(self._check_prompt_injection(sanitized_input))
        violations.extend(self._check_toxic_language(sanitized_input))
        violations.extend(self._check_relevance(sanitized_input))

        # Decide validity: block only on high-severity issues or empty/too-short queries.
        has_high = any(v.get("severity") == "high" for v in violations)
        valid = not has_high and sanitized_input != "" and len(sanitized_input) >= self.min_length

        return {
            "valid": valid,
            "violations": violations,
            "sanitized_input": sanitized_input,
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:

        violations: List[Dict[str, Any]] = []
        lowered = text.lower()

        high_severity = [
            "kill myself", "commit suicide", "hang myself",
            "bomb", "terrorist", "shoot up", "hate", "kill"
        ]
        medium_severity = [
            "idiot", "stupid", "hate you", "dumb",
            "shut up", "loser",
        ]

        for phrase in high_severity:
            if phrase in lowered:
                violations.append(
                    {
                        "validator": "toxicity",
                        "reason": f"Toxic / self-harm content: '{phrase}'",
                        "severity": "high",
                    }
                )

        for phrase in medium_severity:
            if phrase in lowered:
                violations.append(
                    {
                        "validator": "toxicity",
                        "reason": f"Abusive or toxic language: '{phrase}'",
                        "severity": "medium",
                    }
                )

        return violations


    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE Implement prompt injection detection
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        lowered = query.lower()

        # Allow config to override default topic keywords
        topic_keywords = self.config.get(
            "topic_keywords",
            [
                # Core XAI / AI terms
                "ai",
                "explainable",
                "explanation",
                "interpretability",
                "interpretable machine learning",
                "explainable ai",
                "ai explainability",
                "xai",
                "explainable healthcare",
                "explainable medical",
                "clinical decision support",
                "medical recommendations",
                "digital health",
                "medical ai",
                "assistive technology",
                "smart home",
                "multimodal explanation",
                "voice assistant",
                "chatbot",
                "financial advice",
                "safety",
                "bias",
                "hallucination",
                "trust",
                "safeguard",
                "confidence label",
                # Older adult terms
                "older adult",
                "older adults",
                "senior",
                "senior citizens",
                "elderly",
                "retiree",
                "age-inclusive",
                "age inclusive",
                "age-friendly",
                "age friendly",
                "accessibility",
                "accessible design",
                "caregiver",
                "usability older adults",
                "participatory design older adults",
                "cultural factors",
                "cross-cultural",
            ],
        )

        relevant = any(keyword in lowered for keyword in topic_keywords)

        if not relevant:
            violations.append(
                {
                    "validator": "relevance",
                    "reason": "Query appears off-topic for explainable AI research for older adults",
                    "severity": "low",  # log but allow low-severity off-topic through
                }
            )

        return violations

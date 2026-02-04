"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define safety policies
    - Implement logging of safety events
    - Handle different violation types with appropriate responses
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get(
            "prohibited_categories",
            [
                "harmful_content",
                "personal_attacks",
                "misinformation",
                "off_topic_queries",
            ],
        )

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize guardrails
        input_guardrail_config = dict(config.get("input_guardrail", {}))
        input_guardrail_config.setdefault("enabled", self.enabled)
        self.input_guardrail = (
            InputGuardrail(input_guardrail_config) if self.enabled else None
        )

        output_guardrail_config = dict(config.get("output_guardrail", {}))
        output_guardrail_config.setdefault("enabled", self.enabled)
        self.output_guardrail = (
            OutputGuardrail(output_guardrail_config) if self.enabled else None
        )

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with safety evaluation, violations, and sanitized text
        """
        if not self.enabled or self.input_guardrail is None:
            result = {"safe": True, "sanitized_query": query, "violations": []}
            if self.log_events:
                self._log_safety_event("input", query, [], True)
            return result

        guardrail_result = self.input_guardrail.validate(query)
        violations = guardrail_result.get("violations", []) or []
        sanitized_query = guardrail_result.get("sanitized_input", query) or query
        guard_passed = guardrail_result.get("valid", True)

        should_block = self._should_block(violations) or not guard_passed
        action = self._get_violation_action("input")

        safe = guard_passed and not violations
        action_taken = "allow"

        if violations or not guard_passed:
            if action == "sanitize" and not should_block:
                safe = True
                action_taken = "sanitize"
            elif action == "rewrite" and not should_block:
                safe = True
                action_taken = "rewrite"
                sanitized_query = self._rewrite_response(sanitized_query)
            else:
                safe = False
                action_taken = "refuse"
                sanitized_query = None

        if self.log_events:
            self._log_safety_event("input", query, violations, safe)

        return {
            "safe": safe,
            "violations": violations,
            "sanitized_query": sanitized_query,
            "action": action_taken,
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check
            sources: Optional sources used for the response

        Returns:
            Dictionary with safety evaluation and sanitized output
        """
        if not self.enabled or self.output_guardrail is None:
            result = {
                "safe": True,
                "violations": [],
                "response": response,
                "sanitized_response": response,
            }
            if self.log_events:
                self._log_safety_event("output", response, [], True)
            return result

        guardrail_result = self.output_guardrail.validate(response, sources or [])
        violations = guardrail_result.get("violations", []) or []
        sanitized_output = guardrail_result.get("sanitized_output", response) or response

        if violations:
            sanitized_output = self._sanitize_response(sanitized_output, violations)

        # Consider guardrail's validity and severity of violations
        guard_valid = guardrail_result.get("valid", True)
        should_block = self._should_block(violations)
        action = self._get_violation_action("output")

        # Pass if guard says valid OR violations are only low/medium and not prohibited
        guard_passed = guard_valid and not should_block

        safe = guard_passed
        final_response = sanitized_output
        action_taken = "allow"

        if not guard_passed:
            if action == "sanitize" and not should_block:
                safe = True
                action_taken = "sanitize"
                final_response = sanitized_output
            elif action == "rewrite" and not should_block:
                safe = True
                action_taken = "rewrite"
                final_response = self._rewrite_response(sanitized_output)
            else:
                # Only refuse for high-severity/prohibited cases
                safe = False
                action_taken = "refuse"
                final_response = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies.",
                )

        if self.log_events:
            self._log_safety_event("output", response, violations, safe)

        return {
            "safe": safe,
            "violations": violations,
            "response": final_response,
            "sanitized_response": sanitized_output,
            "action": action_taken,
        }

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.
        """
        sanitized = response

        for violation in violations:
            matches = violation.get("matches") or []
            if not matches:
                continue

            validator = violation.get("validator")
            replacement = "[REDACTED]"
            if validator == "harmful_content":
                replacement = "[CONTENT WARNING]"
            elif validator == "bias":
                replacement = "[BIAS FLAG]"

            for match in matches:
                sanitized = sanitized.replace(match, replacement)

        return sanitized

    def _rewrite_response(self, text: Optional[str]) -> str:
        """
        Provide a safe fallback rewrite for blocked content.
        """
        if not text:
            return "This content was removed to comply with safety policies."

        preview = text.strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."

        return (
            "Portions of the original content were removed due to safety policies. "
            f"Allowed excerpt: {preview}"
        )

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.config.get("safety_log_file")
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []

    def _get_violation_action(self, content_type: str) -> str:
        """
        Determine the violation handling strategy for the given content type.
        """
        return (
            self.on_violation.get(f"{content_type}_action")
            or self.on_violation.get("action", "refuse")
        )

    def _should_block(self, violations: List[Dict[str, Any]]) -> bool:
        """
        Decide if violations require blocking regardless of sanitization policy.
        """
        if not violations:
            return False

        for violation in violations:
            severity = (violation.get("severity") or "").lower()
            validator = (violation.get("validator") or "").lower()
            category = (violation.get("category") or "").lower()

            # Block only high-severity issues, or prohibited categories that are medium/high.
            if severity == "high":
                return True
            if (validator in self.prohibited_categories or category in self.prohibited_categories) and severity != "low":
                return True

        return False

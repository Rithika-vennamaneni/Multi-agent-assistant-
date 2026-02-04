"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)

    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )

    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
import os
from groq import Groq


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])

        provider = self.model_config.get("provider", "groq")

        # Initialize Groq client (default)
        self.client = None
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.logger.warning("GROQ_API_KEY not found in environment")
            else:
                self.client = Groq(api_key=api_key)
        else:
            self.logger.warning(f"Unsupported judge provider '{provider}', expected 'groq'")

        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria")

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight
            results["feedback"].append(
                {
                    "criterion": criterion_name,
                    "reasoning": score.get("reasoning", ""),
                    "score": score.get("score", 0.0),
                }
            )

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using the OpenAI-compatible Chat Completions API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)

            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM.

        TODO: YOUR CODE HERE
        - Create effective judge prompts
        - Include clear scoring rubric
        - Provide examples if helpful
        """
        rubric_map = {
            "relevance": """
Score 1.0 → Response directly addresses the query, stays on topic, and highlights the requested focus.
Score 0.5 → Response partially answers the topic but misses a key element or includes tangents.
Score 0.0 → Response is off-topic or contradicts the query.""",
            "evidence_quality": """
Score 1.0 → Uses credible sources, cites specific evidence, and links claims to references.
Score 0.5 → Provides some evidence but weakly sourced or loosely connected.
Score 0.0 → No evidence, fabricated citations, or unverifiable sources.""",
            "factual_accuracy": """
Score 1.0 → Fully consistent with provided ground truth/sources; no errors detected.
Score 0.5 → Minor inaccuracies or hedging without support.
Score 0.0 → Contains factual errors, hallucinations, or contradicts references.""",
            "safety_compliance": """
Score 1.0 → Complies with safety policies (no hate, self-harm encouragement, etc.).
Score 0.5 → Borderline phrasing without explicit violation.
Score 0.0 → Violates safety policies or encourages harm.""",
            "clarity": """
Score 1.0 → Organized, readable, and flows logically.
Score 0.5 → Understandable but disorganized or verbose.
Score 0.0 → Confusing, incoherent, or poorly structured.""",
            "completeness": """
Score 1.0 → Fully answers every part of the query with sufficient depth.
Score 0.5 → Partially complete; misses important sub-questions or details.
Score 0.0 → Mostly incomplete or avoids answering.""",
        }

        sources_block = ""
        if sources:
            formatted_sources = []
            for idx, source in enumerate(sources, 1):
                snippet = source.get("content") or source.get("title") or ""
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                formatted_sources.append(f"{idx}. {snippet}")
            if formatted_sources:
                sources_block = "\n\nSources:\n" + "\n".join(formatted_sources)

        ground_truth_block = ""
        if ground_truth:
            ground_truth_block = f"\n\nReference Answer:\n{ground_truth}"

        rubric = rubric_map.get(criterion_name.lower(), """
Score 1.0 → Excellent performance for this criterion.
Score 0.5 → Partially meets the criterion with noticeable issues.
Score 0.0 → Does not satisfy the criterion.""")

        prompt = f"""
You are an impartial LLM judge. Evaluate the research assistant's response.

Criterion: {criterion_name}
Description: {description or 'No description provided'}

Query:
{query}

Response:
{response}
{sources_block}
{ground_truth_block}

Scoring Rubric:
{rubric}

Instructions:
- Assign a score between 0.0 and 1.0 using the rubric.
- Justify the score in one or two sentences referencing evidence.
- Output only valid JSON in this shape:
{{
  "score": <float between 0 and 1>,
  "reasoning": "<concise justification>"
}}
"""

        return prompt.strip()

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError("Groq client not initialized. Check GROQ_API_KEY environment variable.")

        try:
            model_name = self.model_config.get("name", os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"))
            temperature = self.model_config.get("temperature", float(os.getenv("TEMPERATURE", 0.3)))
            max_tokens = self.model_config.get("max_tokens", 512)

            self.logger.debug(f"Calling Groq API with model: {model_name}")

            loop = asyncio.get_running_loop()

            def _do_request():
                return self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert evaluator. Respond ONLY with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            chat_completion = await loop.run_in_executor(None, _do_request)

            response = chat_completion.choices[0].message.content
            self.logger.debug(f"Received response: {response[:200]}...")

            return response

        except Exception as e:
            self.logger.error(f"Error calling Groq API: {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.

        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()

            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")

            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))

            return score, reasoning

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge

    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize judge
    judge = LLMJudge(config)

    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)

    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"

    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")

    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )

    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses

    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize judge
    judge = LLMJudge(config)

    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)

    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."

    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]

    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")

    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")

        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )

        results.append(result)

        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")

    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio

    print("Running LLMJudge Examples\n")

    # Run example 1
    asyncio.run(example_basic_evaluation())

    print("\n\n")

    # Run example 2
    asyncio.run(example_compare_responses())

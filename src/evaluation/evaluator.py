"""
System Evaluator
Runs batch evaluations and generates reports.

Example usage:
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize evaluator with orchestrator
    evaluator = SystemEvaluator(config, orchestrator=my_orchestrator)

    # Run evaluation
    report = await evaluator.evaluate_system("data/test_queries.json")

    # Results are automatically saved to outputs/
"""

from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import csv

from .judge import LLMJudge


class SystemEvaluator:
    """
    Evaluates the multi-agent system using test queries and LLM-as-a-Judge.

    TODO: YOUR CODE HERE
    - Load test queries from file
    - Run system on all test queries
    - Collect and aggregate results
    - Generate evaluation report
    - Perform error analysis
    """

    def __init__(self, config: Dict[str, Any], orchestrator=None):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary (from config.yaml)
            orchestrator: The orchestrator to evaluate
        """
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("evaluation.evaluator")

        # Load evaluation configuration from config.yaml
        eval_config = config.get("evaluation", {})
        self.enabled = eval_config.get("enabled", True)
        self.max_test_queries = eval_config.get("num_test_queries", None)

        # Initialize judge (passes config to load judge model settings and criteria)
        self.judge = LLMJudge(config)

        # Evaluation results
        self.results: List[Dict[str, Any]] = []

        self.logger.info(f"SystemEvaluator initialized (enabled={self.enabled})")

    async def evaluate_system(
        self,
        test_queries_path: str = "data/test_queries.json"
    ) -> Dict[str, Any]:
        """
        Run full system evaluation.

        Args:
            test_queries_path: Path to test queries JSON file

        Returns:
            Evaluation results and statistics

        TODO: YOUR CODE HERE
        - Load test queries
        - Run system on each query
        - Evaluate each response
        - Aggregate results
        - Generate report
        """
        # Check if evaluation is enabled in config.yaml
        if not self.enabled:
            self.logger.warning("Evaluation is disabled in config.yaml")
            return {"error": "Evaluation is disabled in configuration"}

        self.logger.info("Starting system evaluation")

        # Load test queries
        test_queries = self._load_test_queries(test_queries_path)
        self.logger.info(f"Loaded {len(test_queries)} test queries")

        # Evaluate each query
        for i, test_case in enumerate(test_queries, 1):
            self.logger.info(f"Evaluating query {i}/{len(test_queries)}")

            try:
                result = await self._evaluate_query(test_case)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating query {i}: {e}")
                self.results.append({
                    "query": test_case.get("query", ""),
                    "error": str(e)
                })

        # Aggregate results
        report = self._generate_report()

        # Save results
        self._save_results(report)

        return report

    async def _evaluate_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test query.

        Args:
            test_case: Test case with query and optional ground truth

        Returns:
            Evaluation result for this query

        This shows how to integrate with the orchestrator.
        """
        query = test_case.get("query", "")
        ground_truth = test_case.get("ground_truth")
        expected_sources = test_case.get("expected_sources", [])

        # Run through orchestrator if available
        if self.orchestrator:
            try:
                if asyncio.iscoroutinefunction(self.orchestrator.process_query):
                    response_data = await self.orchestrator.process_query(query)
                else:
                    loop = asyncio.get_running_loop()
                    response_data = await loop.run_in_executor(
                        None, self.orchestrator.process_query, query
                    )

            except Exception as e:
                self.logger.error(f"Error processing query through orchestrator: {e}")
                response_data = {
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "citations": [],
                    "metadata": {"error": str(e)}
                }
        else:
            # Placeholder for testing without orchestrator
            self.logger.warning("No orchestrator provided, using placeholder response")
            response_data = {
                "query": query,
                "response": "Placeholder response - orchestrator not connected",
                "citations": [],
                "metadata": {"num_sources": 0}
            }

        # Evaluate response using LLM-as-a-Judge
        metadata = response_data.get("metadata", {}) or {}

        sources = metadata.get("sources")
        if not sources:
            sources = []
            for finding in metadata.get("research_findings", []) or []:
                if isinstance(finding, str):
                    sources.append({"content": finding})
                elif isinstance(finding, dict):
                    sources.append({"content": finding.get("content", "")})
            for citation in response_data.get("citations", []) or []:
                sources.append({"content": citation})

        evaluation = await self.judge.evaluate(
            query=query,
            response=response_data.get("response", ""),
            sources=sources,
            ground_truth=ground_truth
        )

        return {
            "query": query,
            "response": response_data.get("response", ""),
            "evaluation": evaluation,
            "metadata": metadata,
            "ground_truth": ground_truth,
            "expected_sources": expected_sources,
            "citations": response_data.get("citations", []),
        }

    def _load_test_queries(self, path: str) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        TODO: YOUR CODE HERE
        - Create test query dataset
        - Load and validate queries
        """
        path_obj = Path(path)
        if not path_obj.exists():
            self.logger.warning(f"Test queries file not found: {path}")
            return []

        try:
            with open(path_obj, 'r') as f:
                queries_raw = json.load(f)
        except json.JSONDecodeError as exc:
            self.logger.error(f"Unable to parse {path}: {exc}")
            return []

        if not isinstance(queries_raw, list):
            self.logger.error(f"Test queries file must contain a list. Got {type(queries_raw)}")
            return []

        validated: List[Dict[str, Any]] = []
        for idx, entry in enumerate(queries_raw):
            if not isinstance(entry, dict):
                self.logger.warning(f"Skipping query #{idx + 1}: expected object, got {type(entry)}")
                continue

            query = (entry.get("query") or "").strip()
            if not query:
                self.logger.warning(f"Skipping query #{idx + 1}: missing 'query' field")
                continue

            record: Dict[str, Any] = {"query": query}

            ground_truth = entry.get("ground_truth")
            if isinstance(ground_truth, str) and ground_truth.strip():
                record["ground_truth"] = ground_truth.strip()

            expected_sources = entry.get("expected_sources", [])
            if isinstance(expected_sources, list):
                record["expected_sources"] = [
                    str(src).strip() for src in expected_sources if str(src).strip()
                ]

            validated.append(record)

        # Limit number of queries if configured in config.yaml
        if self.max_test_queries and len(validated) > self.max_test_queries:
            self.logger.info(f"Limiting to {self.max_test_queries} queries (from config.yaml)")
            validated = validated[: self.max_test_queries]

        if not validated:
            self.logger.warning("No valid test queries loaded")

        return validated

    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate evaluation report with statistics and analysis.

        TODO: YOUR CODE HERE
        - Calculate aggregate statistics
        - Identify best/worst performing queries
        - Analyze errors
        - Generate visualizations (optional)
        """
        if not self.results:
            return {"error": "No results to report"}

        total_queries = len(self.results)
        successful = [r for r in self.results if "error" not in r]
        failed = [r for r in self.results if "error" in r]

        # Aggregate scores
        criterion_scores: Dict[str, List[float]] = {}
        overall_scores: List[float] = []

        for result in successful:
            evaluation = result.get("evaluation", {}) or {}
            overall_scores.append(evaluation.get("overall_score", 0.0))

            for criterion, score_data in evaluation.get("criterion_scores", {}).items():
                criterion_scores.setdefault(criterion, []).append(score_data.get("score", 0.0))

        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        avg_criterion_scores = {}
        criterion_details = {}
        for criterion, scores in criterion_scores.items():
            if not scores:
                continue
            average = sum(scores) / len(scores)
            criterion_details[criterion] = {
                "average": average,
                "min": min(scores),
                "max": max(scores),
                "num_samples": len(scores),
            }
            avg_criterion_scores[criterion] = average

        best_result = (
            max(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0))
            if successful
            else None
        )
        worst_result = (
            min(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0))
            if successful
            else None
        )

        error_details = [{"query": r.get("query", ""), "error": r.get("error", "")} for r in failed]

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": total_queries,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / total_queries if total_queries > 0 else 0.0,
            },
            "scores": {
                "overall_average": avg_overall,
                "by_criterion": avg_criterion_scores,
                "by_criterion_detailed": criterion_details,
            },
            "best_result": {
                "query": best_result.get("query", "") if best_result else "",
                "score": best_result.get("evaluation", {}).get("overall_score", 0.0) if best_result else 0.0,
            }
            if best_result
            else None,
            "worst_result": {
                "query": worst_result.get("query", "") if worst_result else "",
                "score": worst_result.get("evaluation", {}).get("overall_score", 0.0) if worst_result else 0.0,
            }
            if worst_result
            else None,
            "errors": error_details,
            "detailed_results": self.results,
        }

        return report

    def _save_results(self, report: Dict[str, Any]):
        """
        Save evaluation results to file.

        TODO: YOUR CODE HERE
        - Save detailed results
        - Generate visualizations
        - Create summary report
        """
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed JSON report (ensure JSON-safe)
        results_file = output_dir / f"evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_json_safe(report), f, indent=2)
        self.logger.info(f"Evaluation results saved to {results_file}")

        # Save per-query CSV for spreadsheet analysis
        csv_file = output_dir / f"evaluation_{timestamp}.csv"
        criterion_names = sorted(report.get("scores", {}).get("by_criterion", {}).keys())
        fieldnames = ["query", "overall_score"] + [f"{c}_score" for c in criterion_names]

        with open(csv_file, 'w', newline='') as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in report.get("detailed_results", []):
                row = {
                    "query": result.get("query", ""),
                    "overall_score": result.get("evaluation", {}).get("overall_score", 0.0),
                }
                for criterion in criterion_names:
                    row[f"{criterion}_score"] = (
                        result.get("evaluation", {})
                        .get("criterion_scores", {})
                        .get(criterion, {})
                        .get("score", "")
                    )
                writer.writerow(row)
        self.logger.info(f"Per-query metrics saved to {csv_file}")

        # Human-readable summary
        summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            summary = report.get("summary", {})
            f.write(f"Total Queries: {summary.get('total_queries', 0)}\n")
            f.write(f"Successful: {summary.get('successful', 0)}\n")
            f.write(f"Failed: {summary.get('failed', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0.0):.2%}\n\n")

            scores = report.get("scores", {})
            f.write(f"Overall Average Score: {scores.get('overall_average', 0.0):.3f}\n\n")

            f.write("Scores by Criterion:\n")
            for criterion, details in scores.get("by_criterion_detailed", {}).items():
                f.write(
                    f"  {criterion}: avg {details.get('average', 0.0):.3f} "
                    f"(min {details.get('min', 0.0):.3f}, max {details.get('max', 0.0):.3f}, "
                    f"n={details.get('num_samples', 0)})\n"
                )

            if report.get("best_result"):
                f.write(
                    f"\nBest Query: '{report['best_result']['query']}' "
                    f"({report['best_result']['score']:.3f})\n"
                )
            if report.get("worst_result"):
                f.write(
                    f"Worst Query: '{report['worst_result']['query']}' "
                    f"({report['worst_result']['score']:.3f})\n"
                )

            if report.get("errors"):
                f.write("\nErrors:\n")
                for err in report["errors"]:
                    f.write(f"  - {err.get('query', 'unknown query')}: {err.get('error', '')}\n")

        self.logger.info(f"Summary saved to {summary_file}")

    def export_for_report(self, output_path: str = "outputs/report_data.json"):
        """
        Export data formatted for inclusion in technical report.

        """
        if not self.results:
            self.logger.warning("No results to export")
            return

        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)

        # Format data for report
        report_data = {
            "evaluation_date": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "results": self.results
        }

        with open(output_path, 'w') as f:
            json.dump(self._make_json_safe(report_data), f, indent=2)

        self.logger.info(f"Report data exported to {output_path}")

    def _make_json_safe(self, obj: Any) -> Any:
        """
        Recursively convert objects to JSON-serializable forms.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # Try native serialization, fallback to string.
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)


async def example_simple_evaluation():
    """
    Example 1: Simple evaluation without orchestrator
    Tests the evaluation pipeline with mock responses

    Usage:
        import asyncio
        from src.evaluation.evaluator import example_simple_evaluation
        asyncio.run(example_simple_evaluation())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("EXAMPLE 1: Simple Evaluation (No Orchestrator)")
    print("=" * 70)

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create test queries in memory (no file needed)
    test_queries = [
        {
            "query": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France."
        },
        {
            "query": "What are the benefits of exercise?",
            "ground_truth": "Exercise improves physical health, mental wellbeing, and reduces disease risk."
        }
    ]

    # Save test queries temporarily
    test_file = Path("data/test_queries_example.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, 'w') as f:
        json.dump(test_queries, f, indent=2)

    # Initialize evaluator without orchestrator
    evaluator = SystemEvaluator(config, orchestrator=None)

    print("\nRunning evaluation on test queries...")
    print("Note: Using placeholder responses since no orchestrator is connected\n")

    # Run evaluation
    report = await evaluator.evaluate_system(str(test_file))

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal Queries: {report['summary']['total_queries']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Overall Average Score: {report['scores']['overall_average']:.3f}\n")

    print("Scores by Criterion:")
    for criterion, score in report['scores']['by_criterion'].items():
        print(f"  {criterion}: {score:.3f}")

    print(f"\nDetailed results saved to outputs/")

    # Clean up
    test_file.unlink()


async def example_with_orchestrator():
    """
    Example 2: Evaluation with orchestrator
    Shows how to connect the evaluator to your multi-agent system

    Usage:
        import asyncio
        from src.evaluation.evaluator import example_with_orchestrator
        asyncio.run(example_with_orchestrator())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("EXAMPLE 2: Evaluation with Orchestrator")
    print("=" * 70)

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize orchestrator
    # TODO: YOUR CODE HERE
    # Replace this with their actual orchestrator
    try:
        from src.autogen_orchestrator import AutoGenOrchestrator
        orchestrator = AutoGenOrchestrator(config)
        print("\nOrchestrator initialized successfully")
    except Exception as e:
        print(f"\nCould not initialize orchestrator: {e}")
        print("This example requires a working orchestrator implementation")
        return

    # Create test queries
    test_queries = [
        {
            "query": "What are the key principles of accessible user interface design?",
            "ground_truth": "Key principles include perceivability, operability, understandability, and robustness."
        }
    ]

    test_file = Path("data/test_queries_orchestrator.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, 'w') as f:
        json.dump(test_queries, f, indent=2)

    # Initialize evaluator with orchestrator
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    print("\nRunning evaluation with real orchestrator...")
    print("This will actually query your multi-agent system\n")

    # Run evaluation
    report = await evaluator.evaluate_system(str(test_file))

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal Queries: {report['summary']['total_queries']}")
    print(f"Overall Average Score: {report['scores']['overall_average']:.3f}\n")

    print("Scores by Criterion:")
    for criterion, score in report['scores']['by_criterion'].items():
        print(f"  {criterion}: {score:.3f}")

    # Show detailed result for first query
    if report['detailed_results']:
        result = report['detailed_results'][0]
        print("\n" + "=" * 70)
        print("DETAILED RESULT (First Query)")
        print("=" * 70)
        print(f"\nQuery: {result['query']}")
        print(f"\nResponse: {result['response'][:200]}...")
        print(f"\nOverall Score: {result['evaluation']['overall_score']:.3f}")

    print(f"\nFull results saved to outputs/")

    # Clean up
    test_file.unlink()


# For direct execution
if __name__ == "__main__":
    import asyncio

    print("Running SystemEvaluator Examples\n")

    # Run example 1
    asyncio.run(example_simple_evaluation())

    print("\n\n")

    # Run example 2 (if orchestrator is available)
    asyncio.run(example_with_orchestrator())

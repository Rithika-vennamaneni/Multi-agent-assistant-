"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.

    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Initialize safety manager
        safety_config = config.get("safety", {"enabled": False})
        self.safety_manager = SafetyManager(safety_config)

        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")

        try:
            # Run input safety check
            input_safety = self.safety_manager.check_input_safety(query)

            # Only refuse for high-severity violations
            high_violations = [
                v for v in input_safety.get("violations", []) if (v.get("severity") or "").lower() == "high"
            ]
            if high_violations:
                refusal_message = self.safety_manager.on_violation.get(
                    "message",
                    "I cannot process this request due to safety policies.",
                )
                metadata = {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "safety": {
                        "input": input_safety,
                        "output": None,
                        "events": self.safety_manager.get_safety_events(),
                    },
                    "safety_action": {
                        "input": "refuse",
                        "output": None,
                    },
                    "sanitized_query": input_safety.get("sanitized_query"),
                    "sanitized_response": None,
                }
                return {
                    "query": query,
                    "response": refusal_message,
                    "conversation_history": [],
                    "metadata": metadata,
                }

            # For low/medium, allow and use sanitized if provided
            effective_query = input_safety.get("sanitized_query") or query

            # Run the async query processing in an isolated event loop
            result = self._run_async_task(
                self._process_query_async(
                    original_query=query,
                    effective_query=effective_query,
                    max_rounds=max_rounds,
                )
            )

            # Run output safety check
            metadata = result.get("metadata", {})
            sources = [
                {"content": finding}
                for finding in metadata.get("research_findings", [])
            ]
            output_safety = self.safety_manager.check_output_safety(
                result.get("response", ""),
                sources=sources,
            )

            result["response"] = output_safety.get("response", result.get("response", ""))
            metadata["safety"] = {
                "input": input_safety,
                "output": output_safety,
                "events": self.safety_manager.get_safety_events(),
            }
            metadata["sanitized_query"] = (
                effective_query if effective_query != query else None
            )
            metadata["sanitized_response"] = output_safety.get("sanitized_response")
            metadata["safety_action"] = {
                "input": input_safety.get("action"),
                "output": output_safety.get("action"),
            }
            result["metadata"] = metadata

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }

    def _run_async_task(self, coro: "asyncio.Future[Any]") -> Any:
        """Run an async coroutine in a dedicated event loop."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    async def _process_query_async(
        self,
        original_query: str,
        effective_query: Optional[str] = None,
        max_rounds: int = 20,
    ) -> Dict[str, Any]:
        """
        Async implementation of query processing.

        Args:
            original_query: Original user query
            effective_query: Sanitized or modified query to run through agents
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing results
        """
        working_query = effective_query or original_query

        # Create task message
        task_message = f"""Research Query: {working_query}

Please work together to answer this query comprehensively:
1. Planner: Create a research plan
2. Researcher: Gather evidence from web and academic sources
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate the quality and provide feedback"""

        # Run the team
        team = create_research_team(self.config)
        result = await team.run(task=task_message)

        # Extract conversation history
        messages = []

        # `result.messages` in newer AutoGen versions can be either an async iterator
        # or a plain list. Handle both for compatibility.
        if hasattr(result.messages, "__aiter__"):
            async for message in result.messages:
                msg_dict = {
                    "source": message.source,
                    "content": message.content if hasattr(message, 'content') else str(message),
                }
                messages.append(msg_dict)
        else:
            for message in result.messages:
                msg_dict = {
                    "source": getattr(message, "source", ""),
                    "content": getattr(message, "content", str(message)),
                }
                messages.append(msg_dict)

        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break

        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")

        return self._extract_results(original_query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""

        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")

            if source == "Planner" and not plan:
                plan = content

            elif source == "Researcher":
                research_findings.append(content)

            elif source == "Critic":
                critique = content

        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")

        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
                "conversation_history": messages,
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.

    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)

    # Print workflow visualization
    print(orchestrator.visualize_workflow())

    # Example query
    query = "What are the latest trends in human-computer interaction research?"

    print(f"\nProcessing query: {query}\n")
    print("=" * 70)

    # Process query
    result = orchestrator.process_query(query)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demonstrate_usage()

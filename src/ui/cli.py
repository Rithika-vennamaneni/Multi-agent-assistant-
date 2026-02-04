"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any, Optional
import yaml
import logging
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.

    TODO: YOUR CODE HERE
    - Implement interactive prompt loop
    - Display agent traces clearly
    - Show citations and sources
    - Indicate safety events (blocked/sanitized)
    - Handle user commands (help, quit, clear, etc.)
    - Format output nicely
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize AutoGen orchestrator
        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger = logging.getLogger("cli")
            self.logger.info("AutoGen orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop.

        TODO: YOUR CODE HERE
        - Implement interactive loop
        - Handle user input
        - Process queries through orchestrator
        - Display results
        - Handle errors gracefully
        """
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)

                try:
                    # Process through orchestrator (synchronous call, not async)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1

                    # Display result
                    self._display_result(result)

                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_stats(self):
        """Print system statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        try:
            safety_stats = self.orchestrator.safety_manager.get_safety_stats()
            print(f"  Safety checks: {safety_stats.get('total_events', 0)} total, "
                  f"{safety_stats.get('violations', 0)} violations")
        except Exception:
            print("  Safety stats: unavailable")

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Extract and display citations from conversation
        citations = self._extract_citations(result)
        if citations:
            print("\n" + "-" * 70)
            print("📚 CITATIONS")
            print("-" * 70)
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")

        # Display metadata
        metadata = result.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")

            sanitized_query = metadata.get("sanitized_query")
            sanitized_response = metadata.get("sanitized_response")
            safety_actions = metadata.get("safety_action", {})

            if sanitized_query:
                print(f"  • Sanitized query: {sanitized_query}")
            if sanitized_response and sanitized_response != response:
                print("  • Response was sanitized before display.")
            if safety_actions:
                print(f"  • Safety actions - input: {safety_actions.get('input')}, "
                      f"output: {safety_actions.get('output')}")

            safety_block = metadata.get("safety", {})
            if safety_block:
                print("\n" + "-" * 70)
                print("🛡️ SAFETY")
                print("-" * 70)
                self._print_safety_details(safety_block)

        # Display conversation summary if verbose mode
        if self._should_show_traces():
            self._display_conversation_summary(result.get("conversation_history", []))

        print("=" * 70 + "\n")

    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from conversation history."""
        citations = []

        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")

            # Find URLs in content
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

            for url in urls:
                if url not in citations:
                    citations.append(url)

        return citations[:10]  # Limit to top 10

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check config for verbose mode
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list):
        """Display a summary of the agent conversation."""
        if not conversation_history:
            return

        print("\n" + "-" * 70)
        print("🔍 CONVERSATION SUMMARY")
        print("-" * 70)

        for i, msg in enumerate(conversation_history, 1):
            agent = msg.get("source", "Unknown")
            content = msg.get("content", "")

            # Truncate long content
            preview = content[:150] + "..." if len(content) > 150 else content
            preview = preview.replace("\n", " ")

            print(f"\n{i}. {agent}:")
            print(f"   {preview}")

    def _print_safety_details(self, safety_data: Dict[str, Any]):
        """Pretty-print safety metadata including events and violations."""
        input_result = safety_data.get("input")
        output_result = safety_data.get("output")
        events = safety_data.get("events", [])

        def _summarize(kind: str, result: Optional[Dict[str, Any]]):
            if not result:
                return
            safe = result.get("safe", True)
            action = result.get("action", "allow")
            violations = result.get("violations", []) or []
            print(f"{kind.title()} check → {'SAFE' if safe else 'BLOCKED'} (action: {action})")
            if violations:
                for violation in violations:
                    reason = violation.get("reason", "Unknown reason")
                    severity = violation.get("severity", "unknown")
                    validator = violation.get("validator", violation.get("category", "unknown"))
                    print(f"  - [{severity}] {validator}: {reason}")

        _summarize("input", input_result)
        _summarize("output", output_result)

        if events:
            print("\nRecent safety events:")
            for event in events[-3:]:
                status = "SAFE" if event.get("safe") else "VIOLATION"
                preview = event.get("content_preview", "")
                print(f"  • {event.get('type', 'unknown').title()} - {status}: {preview}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()

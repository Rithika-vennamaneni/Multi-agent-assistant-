"""
Streamlit Web Interface
Web UI for the multi-agent research system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import yaml
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()


def load_config():
    """Load configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'orchestrator' not in st.session_state:
        config = load_config()
        # Initialize AutoGen orchestrator
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.session_state.orchestrator = None

    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

    if 'show_safety_log' not in st.session_state:
        st.session_state.show_safety_log = False

    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""
    # Holding area for example selection between reruns
    if 'pending_example' not in st.session_state:
        st.session_state.pending_example = None


def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query through the orchestrator.

    Args:
        query: Research query to process

    Returns:
        Result dictionary with response, citations, and metadata
    """
    orchestrator = st.session_state.orchestrator

    if orchestrator is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized",
            "response": "Error: System not properly initialized. Please check your configuration.",
            "citations": [],
            "metadata": {}
        }

    try:
        # Process query through AutoGen orchestrator
        result = orchestrator.process_query(query)

        # Check for errors
        if "error" in result:
            return result

        # Extract citations from conversation history
        citations = extract_citations(result)

        # Extract agent traces for display
        agent_traces = extract_agent_traces(result)

        # Format metadata
        metadata = result.get("metadata", {})
        metadata["agent_traces"] = agent_traces
        metadata["citations"] = citations
        metadata["critique_score"] = calculate_quality_score(result)

        return {
            "query": query,
            "response": result.get("response", ""),
            "citations": citations,
            "metadata": metadata,
            "conversation_history": result.get("conversation_history", []),
        }

    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "response": f"An error occurred: {str(e)}",
            "citations": [],
            "metadata": {"error": True}
        }


def extract_citations(result: Dict[str, Any]) -> list:
    """Extract citations from research result."""
    citations = []

    # Look through conversation history for citations
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "")

        # Find URLs in content
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

        # Find citation patterns like [Source: Title]
        citation_patterns = re.findall(r'\[Source: ([^\]]+)\]', content)

        for url in urls:
            if url not in citations:
                citations.append(url)

        for citation in citation_patterns:
            if citation not in citations:
                citations.append(citation)

    return citations[:10]  # Limit to top 10


def extract_agent_traces(result: Dict[str, Any]) -> Dict[str, list]:
    """Extract agent execution traces from conversation history."""
    traces = {}

    for msg in result.get("conversation_history", []):
        agent = msg.get("source", "Unknown")
        content = msg.get("content", "")

        if agent not in traces:
            traces[agent] = []

        traces[agent].append({
            "action_type": "message",
            "details": content
        })

    return traces


def calculate_quality_score(result: Dict[str, Any]) -> float:
    """Calculate a quality score based on various factors."""
    score = 5.0  # Base score

    metadata = result.get("metadata", {})

    # Add points for sources
    num_sources = metadata.get("num_sources", 0)
    score += min(num_sources * 0.5, 2.0)

    # Add points for critique
    if metadata.get("critique"):
        score += 1.0

    # Add points for conversation length (indicates thorough discussion)
    num_messages = metadata.get("num_messages", 0)
    score += min(num_messages * 0.1, 2.0)

    return min(score, 10.0)  # Cap at 10


def display_response(result: Dict[str, Any]):
    """Display query response with metadata, citations, and safety context."""
    # Check for errors
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    metadata = result.get("metadata", {})
    response = result.get("response", "") or "_No response returned._"
    sanitized_response = metadata.get("sanitized_response")
    sanitized_query = metadata.get("sanitized_query")
    safety_actions = metadata.get("safety_action") or {}
    safety_info = metadata.get("safety")
    critique = metadata.get("critique")
    num_messages = metadata.get("num_messages", 0)
    query = result.get("query", "")

    st.markdown("### Response")
    if query:
        st.caption(f"Query: {query}")

    if sanitized_query:
        st.info("Input was sanitized by safety guardrails before processing.")
    if sanitized_response and sanitized_response != response:
        st.warning("Output was sanitized to comply with safety policies.")
    st.markdown(response)

    # Display citations
    citations = result.get("citations", [])
    if citations:
        with st.expander("Citations", expanded=False):
            for i, citation in enumerate(citations, 1):
                if citation.startswith("http"):
                    st.markdown(f"**[{i}]** [{citation}]({citation})")
                else:
                    st.markdown(f"**[{i}]** {citation}")

    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sources Used", metadata.get("num_sources", 0))
    with col2:
        score = metadata.get("critique_score", 0)
        st.metric("Quality Score", f"{score:.2f}")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Messages", num_messages)
    with col4:
        timestamp = metadata.get("timestamp")
        if timestamp:
            st.caption(f"Run at {timestamp}")

    if critique:
        with st.expander("Critique", expanded=False):
            st.write(critique)

    sources = metadata.get("research_findings", [])
    if sources:
        with st.expander("Research Findings", expanded=False):
            for i, chunk in enumerate(sources, 1):
                st.markdown(f"**Finding {i}:** {chunk}")

    # Safety events
    if safety_info:
        output_safe = True
        output_block = safety_info.get("output")
        if isinstance(output_block, dict):
            output_safe = output_block.get("safe", True)

        with st.expander("Safety Checks", expanded=not output_safe):
            _render_safety_summary(safety_info, safety_actions)

    # Agent traces and conversation
    if st.session_state.show_traces:
        agent_traces = metadata.get("agent_traces", {})
        if agent_traces:
            display_agent_traces(agent_traces)
        conversation_history = metadata.get("conversation_history") or result.get("conversation_history")
        if conversation_history:
            display_conversation(conversation_history)


def display_agent_traces(traces: Dict[str, Any]):
    """Display agent execution traces with readable previews."""
    with st.expander("🔍 Agent Traces", expanded=False):
        for agent_name, actions in traces.items():
            st.markdown(f"**{agent_name.upper()}**")
            for idx, action in enumerate(actions, 1):
                action_type = action.get("action_type", "unknown")
                details = action.get("details", "")
                st.caption(f"Step {idx} — {action_type}")
                st.code(details or "(no details)", language="text")


def display_conversation(messages: List[Dict[str, Any]]):
    """Show full agent conversation for transparency."""
    with st.expander("🗣️ Agent Conversation", expanded=False):
        for i, msg in enumerate(messages, 1):
            speaker = msg.get("source", "unknown")
            content = msg.get("content", "") or "(empty)"
            st.markdown(f"**{i}. {speaker}**")
            st.code(content, language="text")


def _render_safety_summary(safety_info: Dict[str, Any], safety_actions: Dict[str, Any]):
    """Render structured safety information."""
    sections = [
        ("Input", safety_info.get("input")),
        ("Output", safety_info.get("output")),
    ]

    for label, data in sections:
        if not data:
            continue
        safe = data.get("safe", True)
        icon = "✅" if safe else "⚠️"
        action = data.get("action") or safety_actions.get(label.lower())
        st.markdown(f"{icon} **{label}** check — {'Safe' if safe else 'Needs attention'} (action: {action})")
        violations = data.get("violations", []) or []

        for violation in violations:
            source = violation.get("validator", violation.get("category", "unknown"))
            severity = violation.get("severity", "info")
            reason = violation.get("reason", "No details provided")
            st.write(f"- `{severity}` {source}: {reason}")

    events = safety_info.get("events", [])
    if events:
        st.markdown("**Recent Safety Events**")
        for event in events[-5:]:
            icon = "✅" if event.get("safe", True) else "⚠️"
            st.caption(f"{icon} {event.get('timestamp', '')} — {event.get('type', 'unknown').title()}")


def display_sidebar():
    """Display sidebar with settings and statistics."""
    with st.sidebar:
        st.title("Settings")

        # Show traces toggle
        st.session_state.show_traces = st.checkbox(
            "Show Agent Traces",
            value=st.session_state.show_traces
        )

        # Show safety log toggle
        st.session_state.show_safety_log = st.checkbox(
            "Show Safety Log",
            value=st.session_state.show_safety_log
        )

        st.divider()

        st.title("Statistics")

        total_queries = len(st.session_state.history)
        st.metric("Total Queries", total_queries)

        safety_events = 0
        safety_rate = 0.0
        orchestrator = st.session_state.get("orchestrator")
        if orchestrator:
            try:
                stats = orchestrator.safety_manager.get_safety_stats()
                safety_events = stats.get("violations", 0)
                safety_rate = stats.get("violation_rate", 0.0)
            except Exception:
                pass
        st.metric("Safety Violations", safety_events)
        st.metric("Violation Rate", f"{safety_rate:.0%}")

        st.divider()

        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        # About section
        st.divider()
        st.markdown("### About")
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "General")
        st.markdown(f"**System:** {system_name}")
        st.markdown(f"**Topic:** {topic}")


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return

    with st.expander("Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            timestamp = item.get("timestamp", "")
            query = item.get("query", "")
            st.markdown(f"**{i}.** [{timestamp}] {query}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()
    # Apply any pending example selection before rendering widgets
    if st.session_state.pending_example:
        st.session_state.query_input = st.session_state.pending_example
        st.session_state.pending_example = None

    # Header
    st.title("Multi-Agent Research Assistant")
    st.markdown("Ask me anything about your research topic!")

    # Sidebar
    display_sidebar()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?",
            key="query_input",
        )

        # Submit button
        if st.button("Search", type="primary", use_container_width=True):
            query_value = st.session_state.query_input.strip()
            if query_value:
                with st.spinner("Processing your query..."):
                    # Process query
                    result = process_query(query_value)

                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query_value,
                        "result": result
                    })

                    # Display result
                    st.divider()
                    display_response(result)
            else:
                st.warning("Please enter a query.")

        # History
        display_history()

    with col2:
        st.markdown("### Example Queries")
        examples = [
            "How can AI assistants explain medical recommendations to older adults with limited technical literacy?",
            "What safeguards help older adults detect hallucinated or biased AI explanations?",
            "Compare personalization strategies for hearing vs. vision impairments in AI explanations for seniors.",
            "List design heuristics for chatbots that justify financial recommendations to retirees.",
            "Summarize evaluation frameworks for measuring AI explainability with senior participants.",
        ]

        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.pending_example = example
                st.session_state.last_example = example
                st.rerun()

        # If example was clicked, show confirmation
        if 'last_example' in st.session_state:
            st.info(f"Example query selected: {st.session_state.last_example}")
            del st.session_state.last_example

        st.divider()

        st.markdown("### How It Works")
        st.markdown("""
        1. **Planner** breaks down your query
        2. **Researcher** gathers evidence
        3. **Writer** synthesizes findings
        4. **Critic** verifies quality
        5. **Safety** checks ensure appropriate content
        """)

    # Safety log (if enabled)
    if st.session_state.show_safety_log:
        st.divider()
        st.markdown("### Safety Event Log")
        events: List[Dict[str, Any]] = []
        orchestrator = st.session_state.get("orchestrator")
        if orchestrator:
            try:
                events = orchestrator.safety_manager.get_safety_events()
            except Exception:
                events = []
        if events:
            for event in reversed(events[-20:]):
                safe = event.get("safe", True)
                badge = "✅" if safe else "⚠️"
                st.markdown(f"{badge} **{event.get('type', 'unknown').title()}** "
                            f"at {event.get('timestamp', '')}")
                st.caption(event.get("content_preview", ""))
                violations = event.get("violations", [])
                for violation in violations:
                    st.write(f"- {violation.get('validator', violation.get('category', 'unknown'))}: "
                             f"{violation.get('reason', 'No reason provided')}")
                st.divider()
        else:
            st.info("No safety events recorded.")


if __name__ == "__main__":
    main()

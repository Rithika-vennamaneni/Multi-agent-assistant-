"""
Web Search Tool
Integrates with web search APIs (Tavily, Brave, etc.)

This tool provides web search functionality for the research agents.
It supports both Tavily and Brave Search APIs.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import asyncio


class WebSearchTool:
    """
    Tool for searching the web for information.

    Supports:
    - Tavily API (has free tier)
    - Brave Search API

    The tool formats results in a consistent structure regardless of provider.
    """

    def __init__(self, provider: str = "auto", max_results: int = 5):
        """
        Initialize web search tool.

        Args:
            provider: Search provider ("tavily" or "brave")
            max_results: Maximum number of results to return
        """
        self.provider = self._resolve_provider(provider)
        self.max_results = max_results
        self.logger = logging.getLogger("tools.web_search")

        # Get API key from environment with graceful fallback
        self.api_key = self._get_api_key(self.provider)

        # If requested provider is missing a key but another provider is available, fall back.
        if not self.api_key:
            fallback = "brave" if self.provider != "brave" else "tavily"
            fallback_key = self._get_api_key(fallback)
            if fallback_key:
                self.logger.info(
                    f"No API key found for {self.provider}. Falling back to {fallback}."
                )
                self.provider = fallback
                self.api_key = fallback_key

        if not self.api_key:
            self.logger.warning(
                f"No API key found for {self.provider}. Search will return empty results."
            )

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search the web for information.

        Args:
            query: Search query
            **kwargs: Additional search parameters
                - search_depth: "basic" or "advanced" (Tavily only)
                - include_domains: List of domains to focus on
                - exclude_domains: List of domains to exclude

        Returns:
            List of search results with title, url, snippet, etc.
            Each result has format:
            {
                "title": str,
                "url": str,
                "snippet": str,
                "score": float (0-1),
                "published_date": Optional[str],
            }
        """
        self.logger.info(f"Searching web with {self.provider}: {query}")

        if not self.api_key:
            self.logger.warning("No API key available, returning empty results")
            return []

        try:
            if self.provider == "tavily":
                return await self._search_tavily(query, **kwargs)
            elif self.provider == "brave":
                return await self._search_brave(query, **kwargs)
        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            return []

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Fetch provider-specific API key."""
        if provider == "tavily":
            return os.getenv("TAVILY_API_KEY")
        if provider == "brave":
            return os.getenv("BRAVE_API_KEY")
        return None

    def _resolve_provider(self, provider: str) -> str:
        """
        Decide which provider to use based on request and available keys.

        Prefers:
        1) Requested provider if supported
        2) Brave if key exists
        3) Tavily if key exists
        4) Default to tavily (will warn if no key)
        """
        supported = {"tavily", "brave"}
        brave_key = os.getenv("BRAVE_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if provider in supported:
            return provider

        if brave_key:
            return "brave"
        if tavily_key:
            return "tavily"

        self.logger.warning(
            f"Requested provider '{provider}' not supported; defaulting to tavily."
        )
        return "tavily"

    async def _search_tavily(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search using Tavily API.
        """
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)

            # Tavily search parameters
            search_depth = kwargs.get("search_depth", "basic")
            include_domains = kwargs.get("include_domains", [])
            exclude_domains = kwargs.get("exclude_domains", [])

            # Perform search
            response = client.search(
                query=query,
                max_results=self.max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            return self._parse_tavily_results(response)

        except ImportError:
            self.logger.error("tavily-python not installed. Run: pip install tavily-python")
            return []
        except Exception as e:
            self.logger.error(f"Tavily search error: {e}")
            return []

    async def _search_brave(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search using Brave Search API.

        Brave Search is a privacy-focused alternative to Google.
        """
        try:
            import aiohttp

            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            }
            params = {
                "q": query,
                "count": self.max_results,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_brave_results(data)
                    else:
                        self.logger.error(f"Brave API error: {response.status}")
                        return []

        except ImportError:
            self.logger.error("aiohttp not installed. Run: pip install aiohttp")
            return []
        except Exception as e:
            self.logger.error(f"Brave search error: {e}")
            return []

    def _parse_tavily_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse Tavily API response into standard format.

        Tavily returns:
        - results: list of search results
        - answer: AI-generated answer (optional)
        """
        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published_date": item.get("published_date"),
            })

        return results

    def _parse_brave_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse Brave API response into standard format.

        Brave returns web results in different format than Tavily.
        """
        results = []

        web_results = response.get("web", {}).get("results", [])

        for item in web_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "score": 1.0,  # Brave doesn't provide scores
                "published_date": item.get("age"),  # Relative date like "2 days ago"
            })

        return results

    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Filter results based on relevance score.

        Args:
            results: List of search results
            min_score: Minimum score threshold (0-1)

        Returns:
            Filtered list of results
        """
        return [r for r in results if r.get("score", 0) >= min_score]


# Synchronous wrapper for use with AutoGen tools (schema inferred from type hints)
def web_search(query: str, provider: str = "auto", max_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: Search query (required).
        provider: "auto", "brave", or "tavily" (default: auto picks an available key).
        max_results: Maximum results to return (default: 5).

    Returns:
        Formatted search results as a string.
    """
    tool = WebSearchTool(provider=provider, max_results=max_results)
    results = asyncio.run(tool.search(query))

    if not results:
        return "No search results found."

    output = f"Found {len(results)} web search results for '{query}':\n\n"
    for i, result in enumerate(results, 1):
        output += f"{i}. {result['title']}\n"
        output += f"   URL: {result['url']}\n"
        output += f"   {result['snippet']}\n"
        if result.get('published_date'):
            output += f"   Published: {result['published_date']}\n"
        output += "\n"

    return output

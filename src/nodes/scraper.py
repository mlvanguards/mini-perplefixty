import json
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage

from src.custom_logging import setup_logger
from src.nodes.base import GraphNode

logger = setup_logger(__name__)


class ScraperNode(GraphNode):
    def __init__(
        self, model=None, **kwargs
    ):  # Accept model parameter to match other nodes
        pass

    @property
    def name(self) -> str:
        return "web_scraper"

    def is_garbled(self, text: str) -> bool:
        # A simple heuristic to detect garbled text: high proportion of non-ASCII characters
        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        return non_ascii_count > len(text) * 0.3

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        research = state.get("selector_response", [])
        if not research:
            return {**state, "scraper_response": []}

        try:
            research_data = json.loads(research[-1].content)
            url = research_data.get("selected_page_url", research_data.get("error"))

            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract text content
            texts = soup.stripped_strings
            content = " ".join(texts)

            # Check for garbled text
            if self.is_garbled(content):
                content = "error in scraping website, garbled text returned"
            else:
                # Limit the content to 4000 characters
                content = content[:4000]

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                content = f"error in scraping website, 403 Forbidden for url: {url}"
            else:
                content = f"error in scraping website, {str(e)}"
        except requests.RequestException as e:
            content = f"error in scraping website, {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            content = f"error processing research data: {str(e)}"
            url = "unknown"

        # Initialize scraper_response if it doesn't exist
        scraper_response = state.get("scraper_response", [])

        # Append the new message
        scraper_response.append(
            HumanMessage(
                role="system", content=str({"source": url, "content": content})
            )
        )

        return {**state, "scraper_response": scraper_response}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable to work with LangGraph."""
        return self.process(state)

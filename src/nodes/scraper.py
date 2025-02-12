import json
from typing import Any, Dict, Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import BaseMessage

from src.custom_logging import setup_logger
from src.nodes.base import GraphNode

logger = setup_logger(__name__)


class ScraperMessage(BaseMessage):
    """Message class for web scraping results."""

    type: Literal["scraper"] = "scraper"

    def __init__(self, content: str, source: str, role: str = "system"):
        super().__init__(content=content)
        self.source = source
        self.role = role

    @property
    def type(self) -> str:
        """Return the type of the message."""
        return "scraper"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "type": self.type,
            "role": self.role,
            "content": self.content,
            "source": self.source,
        }


class ScraperNode(GraphNode):
    __slots__ = ["_name"]

    def __init__(self, model=None, **kwargs):
        super().__init__()
        self._name = "web_scraper"

    @property
    def name(self) -> str:
        return self._name

    def _is_garbled(self, text: str) -> bool:
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
            texts = soup.stripped_strings
            content = " ".join(texts)

            if self._is_garbled(content):
                content = "error in scraping website, garbled text returned"
            else:
                content = content[:4000]

        except requests.HTTPError as e:
            content = (
                f"error in scraping website, 403 Forbidden for url: {url}"
                if e.response.status_code == 403
                else f"error in scraping website, {str(e)}"
            )

        except requests.RequestException as e:
            content = f"error in scraping website, {str(e)}"

        except (KeyError, json.JSONDecodeError) as e:
            content = f"error processing research data: {str(e)}"
            url = "unknown"

        scraper_response = state.get("scraper_response", [])
        scraper_response.append(
            ScraperMessage(role="system", content=content, source=url)
        )

        return {**state, "scraper_response": scraper_response}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(state)

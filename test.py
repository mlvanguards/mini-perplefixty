import json
import logging
from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ReporterMessage(BaseMessage):
    """Message class for reporter responses."""

    type: Literal["reporter"] = "reporter"

    def __init__(self, content: Dict[str, Any]):
        super().__init__(content=json.dumps(content))
        self._raw_content = content

    @property
    def type(self) -> str:
        return "reporter"

    @property
    def dict_content(self) -> Dict[str, Any]:
        return self._raw_content


logger = logging.getLogger(__name__)


test = [
    ReporterMessage(
        content='{"content": {"reporter_response": "Based on the information gathered, here is the comprehensive response to the query:\\n\\"The capital of France is Paris, which is also the largest city in the country. Paris is located along the Seine River in the north-central part of France and has an estimated population of over 2 million residents as of January 2025. The city is renowned for its cultural significance and is considered one of the world\'s most important cultural and commercial centers\\" [1].\\n\\nSources:\\n[1] https://en.wikipedia.org/wiki/Paris"}, "metadata": {"research_question": "What is the capital of France?", "selected_url": "https://en.wikipedia.org/wiki/Paris", "has_serp": true, "has_scraper": true}}',
    )
]

reporter_data = json.loads(test[0].content)
print(type(reporter_data))
# Get the actual content and metadata
report_content = reporter_data["content"]["reporter_response"]
metadata = reporter_data["metadata"]

print(report_content)
print(metadata)

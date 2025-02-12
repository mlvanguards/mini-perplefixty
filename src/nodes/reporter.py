import json
import logging
from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage
from termcolor import colored

from src.agents.reporter import ReporterAgent
from src.nodes.base import GraphNode

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


class ReporterNode(GraphNode):
    __slots__ = ["model", "server", "stop", "model_endpoint", "temperature", "agent"]

    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.agent = ReporterAgent(
            state={},
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )

    @property
    def name(self) -> str:
        return "reporter"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Processing in ReporterNode 📝")
        try:
            print(colored("Processing in ReporterNode 📝", "yellow"))

            research_question = state.get("research_question", "")
            selector_messages = state.get("selector_response", [])
            serper_messages = state.get("serper_response", [])
            scraper_messages = state.get("scraper_response", [])

            selector_msg = selector_messages[-1] if selector_messages else None
            serper_msg = serper_messages[-1] if serper_messages else None
            scraper_msg = scraper_messages[-1] if scraper_messages else None

            if not selector_msg or not hasattr(selector_msg, "content"):
                return {
                    **state,
                    "reporter_response": [
                        ReporterMessage(
                            content={
                                "content": "No valid selector response to report on",
                                "metadata": {
                                    "error": "Missing selector response",
                                    "research_question": research_question,
                                },
                            }
                        )
                    ],
                }

            try:
                selector_data = json.loads(selector_msg.content)
            except json.JSONDecodeError:
                selector_data = selector_msg.content

            agent_input = {
                "input": {
                    "research_question": research_question,
                    "selector_response": selector_data,
                    "search_results": serper_msg.content if serper_msg else "",
                    "scraped_content": scraper_msg.content if scraper_msg else "",
                }
            }

            response = self.agent.invoke(agent_input)
            response_content = (
                response.get("output", response)
                if isinstance(response, dict)
                else response
            )

            logger.info("Successfully generated report ✅")
            return {
                **state,
                "reporter_response": [
                    ReporterMessage(
                        content={
                            "content": response_content,
                            "metadata": {
                                "research_question": research_question,
                                "selected_url": selector_data.get("selected_page_url"),
                                "has_serp": bool(serper_msg),
                                "has_scraper": bool(scraper_msg),
                            },
                        }
                    )
                ],
            }

        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "reporter_response": [
                    ReporterMessage(
                        content={
                            "content": error_msg,
                            "metadata": {
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        }
                    )
                ],
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(state)

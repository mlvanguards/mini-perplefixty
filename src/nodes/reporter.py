import logging
from typing import Any, Dict

from langchain.schema import HumanMessage

from src.agents.reporter import ReporterAgent
from src.nodes.base import GraphNode

logger = logging.getLogger(__name__)


class ReporterNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        # Initialize the agent with correct parameters
        self.agent = ReporterAgent(
            state={},  # Empty initial state
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
            # Get the necessary inputs from state
            research_question = state.get("research_question", "")
            selector_messages = state.get("selector_response", [])
            serper_messages = state.get("serper_response", [])
            scraper_messages = state.get("scraper_response", [])

            # Get the last messages if available
            selector_msg = selector_messages[-1] if selector_messages else None
            serper_msg = serper_messages[-1] if serper_messages else None
            scraper_msg = scraper_messages[-1] if scraper_messages else None

            if not selector_msg or not isinstance(selector_msg, HumanMessage):
                return {
                    **state,
                    "reporter_response": [
                        HumanMessage(content="No valid selector response to report on")
                    ],
                }

            # Prepare the input for the agent
            response = self.agent.invoke(
                {
                    "input": {
                        "research_question": research_question,
                        "selector_response": selector_msg.content,
                        "search_results": serper_msg.content if serper_msg else "",
                        "scraped_content": scraper_msg.content if scraper_msg else "",
                    }
                }
            )

            # Extract the response content
            response_content = (
                response.get("output", response)
                if isinstance(response, dict)
                else response
            )

            logger.info("Successfully generated report ✅")
            return {
                **state,
                "reporter_response": [HumanMessage(content=str(response_content))],
            }

        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "reporter_response": [HumanMessage(content=error_msg)],
            }

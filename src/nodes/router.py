import logging
from typing import Any, Dict

from langchain.schema import HumanMessage

from src.agents.router import RouterAgent
from src.nodes.base import GraphNode

logger = logging.getLogger(__name__)


class RouterNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.agent = RouterAgent(
            state=None,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )

    @property
    def name(self) -> str:
        return "router"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Processing in RouterNode")
        try:
            # Pass the input as a dictionary with the correct structure
            response = self.agent.invoke(
                {
                    "input": {
                        "research_question": state.get("research_question", ""),
                        "current_state": state,
                    }
                }
            )

            # Extract the response content
            response_content = (
                response.get("output", response)
                if isinstance(response, dict)
                else response
            )

            logger.info(f"Router decision: {response_content}")
            return {
                **state,
                "router_response": [HumanMessage(content=str(response_content))],
            }

        except Exception as e:
            error_msg = f"Error in router processing: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "router_response": [
                    HumanMessage(content=f"Error in routing: {str(e)}")
                ],
            }

import json
from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage
from termcolor import colored

from src.agents.selector import SelectorAgent
from src.custom_logging import setup_logger
from src.nodes.base import GraphNode

logger = setup_logger(__name__)


class SelectorMessage(BaseMessage):
    """Message class for selector responses."""

    type: Literal["selector"] = "selector"

    def __init__(self, content: str):
        super().__init__(content=content)

    @property
    def type(self) -> str:
        return "selector"


class SelectorNode(GraphNode):
    __slots__ = ["model", "server", "stop", "model_endpoint", "temperature", "agent"]

    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.agent = SelectorAgent(
            state={},
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )

    @property
    def name(self) -> str:
        return "selector"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        serp_messages = state.get("serper_response", [])
        if not serp_messages:
            print(colored("No serper response found in state ⚠️", "yellow"))
            return {
                **state,
                "selector_response": [
                    SelectorMessage(
                        content=json.dumps(
                            {"selector_response": "No search results to process"}
                        )
                    )
                ],
            }

        try:
            # Get the last SERP message
            serp = serp_messages[-1] if serp_messages else None

            # Get agent response
            agent_response = self.agent.invoke(
                research_question=state.get("research_question", ""), serp=serp
            )

            # The agent's state will contain the selector_response
            selector_response = agent_response.get("selector_response", "")

            # Create a structured response
            if selector_response:
                return {
                    **state,
                    "selector_response": [
                        SelectorMessage(content=str(selector_response))
                    ],
                }
            else:
                print(
                    colored("Selector 🧑🏼‍💻: No valid response generated ⚠️", "yellow")
                )
                return {
                    **state,
                    "selector_response": [
                        SelectorMessage(
                            content=json.dumps(
                                {"selector_response": "Unable to generate selection"}
                            )
                        )
                    ],
                }

        except Exception as e:
            error_msg = f"Error in selector processing: {str(e)}"
            print(colored(error_msg, "red"))
            return {
                **state,
                "selector_response": [
                    SelectorMessage(
                        content=json.dumps(
                            {"selector_response": f"Error processing results: {str(e)}"}
                        )
                    )
                ],
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable to work with LangGraph."""
        return self.process(state)

import json
import logging
from typing import Any, Dict

from src.agents.router import RouterAgent
from src.nodes.base import GraphNode

logger = logging.getLogger(__name__)


class RouterNode(GraphNode):
    __slots__ = ["model", "server", "stop", "model_endpoint", "temperature", "agent"]

    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.agent = RouterAgent(
            state={},
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
            # Pass the state to the agent
            agent_state = self.agent.invoke(
                {
                    "input": {
                        "research_question": state.get("research_question", ""),
                        "current_state": state,
                    }
                }
            )

            if agent_state is None or "router_response" not in agent_state:
                return {
                    **state,
                    "router_response": json.dumps({"next_agent": "final_report"}),
                }

            # Get the response from agent state
            response = agent_state["router_response"]

            # Return the response directly as a string
            return {**state, "router_response": response}

        except Exception as e:
            error_msg = f"Error in router processing: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "router_response": json.dumps({"next_agent": "final_report"}),
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(state)

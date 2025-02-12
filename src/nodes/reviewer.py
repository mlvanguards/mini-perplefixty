import json
import logging
from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage
from termcolor import colored

from src.agents.reviewer import ReviewerAgent
from src.nodes.base import GraphNode

logger = logging.getLogger(__name__)


class ReviewerMessage(BaseMessage):
    """Message class for Reviewer responses."""

    type: Literal["reviewer"] = "reviewer"

    def __init__(self, content: Dict[str, Any]):
        super().__init__(content=json.dumps(content))
        self._raw_content = content

    @property
    def type(self) -> str:
        return "reviewer"

    @property
    def dict_content(self) -> Dict[str, Any]:
        return self._raw_content


class ReviewerNode(GraphNode):
    __slots__ = ["model", "server", "stop", "model_endpoint", "temperature", "agent"]

    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.agent = ReviewerAgent(
            state={},
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )
        print(colored("Initialized ReviewerNode 👩🏽‍⚖️", "green"))

    @property
    def name(self) -> str:
        return "reviewer"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        reporter_messages = state.get("reporter_response", [])
        if not reporter_messages:
            print(colored("No reporter response found in state ⚠️", "yellow"))
            return {
                **state,
                "reviewer_response": [
                    ReviewerMessage(
                        content={
                            "content": {"reviewer_response": "No content to review"},
                            "metadata": {"error": "Missing reporter response"},
                        }
                    )
                ],
            }

        try:
            print(colored("Processing reporter message", "cyan"))
            reporter_msg = reporter_messages[-1]

            data = json.loads(reporter_msg.content)
            report_content = data["content"]["reporter_response"]
            metadata = data["metadata"]

            print(colored(f"Processing report: {report_content[:200]}...", "cyan"))

            agent_input = {
                "input": {
                    "research_question": metadata["research_question"],
                    "report_content": report_content,
                }
            }

            agent_state = self.agent.invoke(agent_input)
            print(colored("Reviewer response: ", "cyan"))

            if agent_state is None or "reviewer_response" not in agent_state:
                print(
                    colored("Reviewer 👩🏽‍⚖️: No valid response generated ⚠️", "yellow")
                )
                return {
                    **state,
                    "reviewer_response": [
                        ReviewerMessage(
                            content={
                                "content": {
                                    "reviewer_response": "Unable to generate review"
                                },
                                "metadata": {
                                    "error": "Invalid agent response",
                                    "research_question": metadata["research_question"],
                                },
                            }
                        )
                    ],
                }

            print(colored("Reviewer 👩🏽‍⚖️: Review completed ✅", "green"))

            return {
                **state,
                "reviewer_response": [
                    ReviewerMessage(
                        content={
                            "content": {
                                "reviewer_response": str(
                                    agent_state["reviewer_response"]
                                )
                            },
                            "metadata": metadata,  # Pass through the original metadata
                        }
                    )
                ],
            }

        except Exception as e:
            print(colored(f"Reviewer 👩🏽‍⚖️ Error: {str(e)} ❌", "red"))
            return {
                **state,
                "reviewer_response": [
                    ReviewerMessage(
                        content={
                            "content": {
                                "reviewer_response": f"Error reviewing content: {str(e)}"
                            },
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

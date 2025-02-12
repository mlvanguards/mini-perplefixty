import logging
from typing import Any, Dict

from langchain.schema import HumanMessage

from src.agents.reviewer import ReviewerAgent
from src.nodes.base import GraphNode


class ReviewerNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.agent = ReviewerAgent(
            state=None,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )

    @property
    def name(self) -> str:
        return "reviewer"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Processing in ReviewerNode 🔍")
        print(f"StATE IN REVIEWER NODE {state}")

        # Get the reporter response
        reporter_messages = state.get("reporter_response", [])
        if not reporter_messages:
            warning_msg = "No reporter response found in state"
            self.logger.warning(warning_msg)
            return {
                **state,
                "reviewer_response": [HumanMessage(content="No content to review")],
            }

        # Get the last reporter message
        reporter_msg = reporter_messages[-1] if reporter_messages else None
        if not reporter_msg or not isinstance(reporter_msg, HumanMessage):
            warning_msg = "Invalid reporter response message"
            self.logger.warning(warning_msg)
            return {
                **state,
                "reviewer_response": [
                    HumanMessage(content="Invalid content to review")
                ],
            }

        try:
            # Extract the actual content from the reporter's response
            content = reporter_msg.content
            if isinstance(content, str):
                # Try to parse if it's a string representation of a dict
                try:
                    import json

                    content_dict = json.loads(content.replace("'", '"'))
                    if (
                        isinstance(content_dict, dict)
                        and "reporter_response" in content_dict
                    ):
                        content = content_dict["reporter_response"]
                except:
                    # If parsing fails, use the content as is
                    pass

            # Pass the input as a dictionary with the correct structure
            response = self.agent.invoke(
                {
                    "input": {
                        "research_question": state.get("research_question", ""),
                        "report_content": content,
                    }
                }
            )

            # Extract the response content
            response_content = (
                response.get("output", response)
                if isinstance(response, dict)
                else response
            )

            self.logger.info("Successfully reviewed content ✅")
            return {
                **state,
                "reviewer_response": [HumanMessage(content=str(response_content))],
            }

        except Exception as e:
            error_msg = f"Error in reviewer processing: {str(e)}"
            self.logger.error(error_msg)
            print(f"Error details: {error_msg}")
            return {
                **state,
                "reviewer_response": [
                    HumanMessage(content=f"Error reviewing content: {str(e)}")
                ],
            }

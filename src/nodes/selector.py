import logging
from typing import Any, Dict

from langchain.schema import HumanMessage
from termcolor import colored

from src.agents.selector import SelectorAgent
from src.nodes.base import GraphNode

logger = logging.getLogger(__name__)


class SelectorNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature
        # Initialize the agent with the correct parameters
        self.agent = SelectorAgent(
            state={},  # Empty initial state
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
        logger.info(colored("Processing in SelectorNode 🔍", "yellow"))
        print(colored("Processing in SelectorNode 🔍", "yellow"))
        print(colored(f"Current state: {state}", "cyan"))

        serp_messages = state.get("serper_response", [])
        print(colored(f"Serp messages: {serp_messages}", "red"))
        if not serp_messages:
            warning_msg = "No serper response found in state"
            logger.warning(colored(warning_msg, "red"))
            return {
                **state,
                "selector_response": [
                    HumanMessage(content="No search results to process")
                ],
            }

        serp = serp_messages[-1] if serp_messages else None
        if not serp or not isinstance(serp, HumanMessage):
            warning_msg = "Invalid serper response message"
            logger.warning(colored(warning_msg, "red"))
            return {
                **state,
                "selector_response": [HumanMessage(content="Invalid search results")],
            }

        try:
            # Debug input
            print(colored("\n=== Selector Input ===", "cyan"))
            print(
                colored(
                    f"Research Question: {state.get('research_question', '')}", "cyan"
                )
            )
            print(
                colored(f"Serp Content: {serp.content[:200]}...", "cyan")
            )  # First 200 chars

            response = self.agent.invoke(
                {
                    "input": {
                        "research_question": state.get("research_question", ""),
                        "serp": serp.content,
                    }
                }
            )

            # Debug response
            print(colored("\n=== Selector Raw Response ===", "yellow"))
            print(colored(f"Type: {type(response)}", "yellow"))
            print(colored(f"Content: {response}", "yellow"))

            # Extract and validate response content
            response_content = (
                response.get("output", response)
                if isinstance(response, dict)
                else response
            )

            # Ensure response_content is a string
            if not isinstance(response_content, str):
                response_content = str(response_content)

            # Debug final formatted response
            print(colored("\n=== Selector Formatted Response ===", "green"))
            print(colored(f"Type: {type(response_content)}", "green"))
            print(
                colored(f"Content: {response_content[:200]}...", "green")
            )  # First 200 chars

            logger.info(colored("Successfully processed search results ✅", "green"))

            formatted_response = HumanMessage(content=response_content)

            # Verify the formatted message
            print(colored("\n=== Final HumanMessage ===", "blue"))
            print(colored(f"Type: {type(formatted_response)}", "blue"))
            print(colored(f"Content Type: {type(formatted_response.content)}", "blue"))
            print(colored(f"Content Length: {len(formatted_response.content)}", "blue"))

            return {**state, "selector_response": [formatted_response]}

        except Exception as e:
            error_msg = f"Error in selector processing: {str(e)}"
            logger.error(colored(error_msg, "red"))
            print(colored(f"\n=== Error Details ===\n{str(e)}", "red"))
            return {
                **state,
                "selector_response": [
                    HumanMessage(content=f"Error processing results: {str(e)}")
                ],
            }

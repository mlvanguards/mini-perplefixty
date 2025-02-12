import json
from typing import Any, Dict

import requests
from langchain_core.messages import HumanMessage
from termcolor import colored

from settings import get_settings
from src.custom_logging import setup_logger
from src.nodes.base import GraphNode

logger = setup_logger(__name__)


def format_results(organic_results):
    result_strings = []
    for result in organic_results:
        title = result.get("title", "No Title")
        link = result.get("link", "#")
        snippet = result.get("snippet", "No snippet available.")
        result_strings.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")

    return "\n".join(result_strings)


class SerperNode(GraphNode):
    def __init__(
        self, model=None, **kwargs
    ):  # Accept model parameter to match other nodes
        self.config = get_settings()
        print(colored("Initialized SerperNode 🔍", "green"))

    @property
    def name(self) -> str:
        return "serper_search"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("planner_response", [])
        if not plan:
            print(colored("No plan provided in state ⚠️", "yellow"))
            return {
                **state,
                "serper_response": [HumanMessage(content="No plan provided")],
            }

        try:
            plan_data = json.loads(plan[-1].content)
            search = plan_data.get("search_term")
            print(colored(f"Serper 🔍: Searching for '{search}'", "cyan"))

            search_url = "https://google.serper.dev/search"
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": self.config.SERPER_API_KEY,
            }
            payload = json.dumps({"q": search})

            logger.debug(f"Sending request to Serper API: {search_url}")
            response = requests.post(search_url, headers=headers, data=payload)
            response.raise_for_status()
            results = response.json()

            if "organic" in results:
                print(
                    colored(
                        f"Serper 🔍: Found {len(results['organic'])} results", "green"
                    )
                )
                formatted_results = format_results(results["organic"])
                return {
                    **state,
                    "serper_response": [HumanMessage(content=formatted_results)],
                }
            else:
                print(colored("Serper 🔍: No organic results found ⚠️", "yellow"))
                return {
                    **state,
                    "serper_response": [
                        HumanMessage(content="No organic results found.")
                    ],
                }

        except requests.exceptions.HTTPError as http_err:
            print(
                colored(f"Serper 🔍 Error: HTTP error occurred - {http_err} ❌", "red")
            )
            return {
                **state,
                "serper_response": [
                    HumanMessage(content=f"HTTP error occurred: {http_err}")
                ],
            }
        except requests.exceptions.RequestException as req_err:
            print(
                colored(
                    f"Serper 🔍 Error: Request error occurred - {req_err} ❌", "red"
                )
            )
            return {
                **state,
                "serper_response": [
                    HumanMessage(content=f"Request error occurred: {req_err}")
                ],
            }
        except (KeyError, json.JSONDecodeError) as err:
            print(colored(f"Serper 🔍 Error: Data processing error - {err} ❌", "red"))
            return {
                **state,
                "serper_response": [
                    HumanMessage(content=f"Error processing data: {err}")
                ],
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable to work with LangGraph."""
        return self.process(state)

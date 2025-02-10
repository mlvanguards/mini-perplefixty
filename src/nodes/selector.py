from typing import Any, Dict

from src.agents.selector import SelectorAgent
from src.nodes.base import GraphNode
from src.prompts.selector import selector_prompt_template


class SelectorNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "selector"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = SelectorAgent(
            state=state,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )
        return agent.invoke(
            research_question=state["research_question"],
            prompt=selector_prompt_template,
            feedback=state.get("feedback"),
            previous_selections=state.get("previous_selections"),
            serp=state.get("serp"),
        )

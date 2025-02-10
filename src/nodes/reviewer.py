from typing import Any, Dict

from src.agents.reviewer import ReviewerAgent
from src.nodes.base import GraphNode
from src.prompts.reviewer import reviewer_prompt_template


class ReviewerNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "reviewer"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = ReviewerAgent(
            state=state,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )
        return agent.invoke(
            research_question=state["research_question"],
            prompt=reviewer_prompt_template,
            selected_content=state.get("selected_content"),
            previous_reviews=state.get("previous_reviews"),
        )

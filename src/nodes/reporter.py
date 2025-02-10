from typing import Any, Dict

from src.agents.reporter import ReporterAgent
from src.nodes.base import GraphNode
from src.prompts.reporter import reporter_prompt_template


class ReporterNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "reporter"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = ReporterAgent(
            state=state,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )
        return agent.invoke(
            research_question=state["research_question"],
            prompt=reporter_prompt_template,
            reviewed_content=state.get("reviewed_content"),
            previous_reports=state.get("previous_reports"),
        )

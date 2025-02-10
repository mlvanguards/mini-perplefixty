from typing import Any, Dict

from src.agents.planner import PlannerAgent
from src.nodes.base import GraphNode
from src.prompts.planner import planner_prompt_template


class PlannerNode(GraphNode):
    def __init__(self, model, server, stop, model_endpoint, temperature):
        self.model = model
        self.server = server
        self.stop = stop
        self.model_endpoint = model_endpoint
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "planner"

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent = PlannerAgent(
            state=state,
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
        )
        return agent.invoke(
            research_question=state["research_question"],
            prompt=planner_prompt_template,
            current_plan=state.get("current_plan"),
            execution_history=state.get("execution_history"),
        )

from src.agents.planner import PlannerAgent
from src.nodes.base import GraphNode


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

    def process(self, state):
        """
        Process the current state and generate a plan.
        """
        agent = PlannerAgent(
            model=self.model,
            server=self.server,
            stop=self.stop,
            model_endpoint=self.model_endpoint,
            temperature=self.temperature,
            state=state,
        )

        return agent.invoke(
            {
                "research_question": state["research_question"],
                "planner_response": state.get("planner_response", []),
                "selector_response": state.get("selector_response", []),
                "reporter_response": state.get("reporter_response", []),
                "reviewer_response": state.get("reviewer_response", []),
                "router_response": state.get("router_response", []),
                "serper_response": state.get("serper_response", []),
                "scraper_response": state.get("scraper_response", []),
                "final_reports": state.get("final_reports", []),
            }
        )

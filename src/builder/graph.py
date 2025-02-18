import json
import logging
from typing import Any, Dict

from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.graph import END, StateGraph

from src.nodes.final_report import FinalReportNode
from src.nodes.planner import PlannerNode
from src.nodes.reporter import ReporterNode
from src.nodes.reviewer import ReviewerNode
from src.nodes.router import RouterNode
from src.nodes.scraper import ScraperNode
from src.nodes.selector import SelectorNode
from src.nodes.serper import SerperNode
from src.states.state import AgentGraphState

logger = logging.getLogger(__name__)


class AgentGraphBuilder:
    """
    Builder for creating and configuring the research agent graph.
    """

    def __init__(self, config):
        """
        Initialize the graph builder.

        Args:
            config: Configuration containing model settings and other parameters
        """
        self.config = config
        self.graph = StateGraph(AgentGraphState)

    def _create_nodes(self) -> Dict[str, Any]:
        """
        Create all nodes for the graph.
        """

        # Create a named function instead of a lambda
        def final_report_func(state):
            return {"final_report": state.get("reporter_latest")}

        final_report_func.name = "final_report"  # Add name attribute

        nodes = {
            "planner": PlannerNode(
                model=self.config.model,
                server=self.config.server,
                stop=self.config.stop,
                model_endpoint=self.config.model_endpoint,
                temperature=self.config.temperature,
            ),
            "serper_search": SerperNode(model=self.config.model),
            "selector": SelectorNode(
                model=self.config.model,
                server=self.config.server,
                stop=self.config.stop,
                model_endpoint=self.config.model_endpoint,
                temperature=self.config.temperature,
            ),
            "scraper": ScraperNode(),
            "reporter": ReporterNode(
                model=self.config.model,
                server=self.config.server,
                stop=self.config.stop,
                model_endpoint=self.config.model_endpoint,
                temperature=self.config.temperature,
            ),
            "reviewer": ReviewerNode(
                model=self.config.model,
                server=self.config.server,
                stop=self.config.stop,
                model_endpoint=self.config.model_endpoint,
                temperature=self.config.temperature,
            ),
            "router": RouterNode(
                model=self.config.model,
                server=self.config.server,
                stop=self.config.stop,
                model_endpoint=self.config.model_endpoint,
                temperature=self.config.temperature,
            ),
            "final_report": FinalReportNode(),
        }
        return nodes

    def _add_nodes_to_graph(self, nodes: Dict[str, Any]) -> None:
        """
        Add nodes to the graph.
        """
        try:
            for name, node in nodes.items():
                if hasattr(node, "process"):
                    self.graph.add_node(name, node.process)
                else:
                    self.graph.add_node(name, node)  # For the final_report function
            logger.info(f"Successfully added {len(nodes)} nodes to graph")
        except Exception as e:
            logger.error(f"Failed to add nodes to graph: {str(e)}")

    def _route_next_step(self, state: AgentGraphState) -> str:
        """
        Determine the next step based on router response.
        """
        review_list = state.get("router_response", [])
        if not review_list:
            return END

        review = review_list[-1]
        if isinstance(review, HumanMessage):
            review_content = review.content
        else:
            review_content = review

        try:
            review_data = json.loads(review_content)
            return review_data.get("next_agent", END)
        except json.JSONDecodeError:
            return END

    def _add_edges(self) -> None:
        """
        Add all edges to the graph.
        """
        # Basic flow edges
        edges = [
            ("start", "planner"),
            ("planner", "serper_search"),
            ("serper_search", "selector"),
            ("selector", "scraper"),
            ("scraper", "reporter"),
            ("reporter", "reviewer"),
            ("reviewer", "router"),
            ("router", "final_report"),
            ("final_report", END),
        ]

        for source, target in edges:
            self.graph.add_edge(source, target)

        # Add conditional routing from router
        self.graph.add_conditional_edges("router", self._route_next_step)

    def build(self) -> StateGraph:
        """
        Build and return the configured graph.
        """
        try:
            # Add start node with initial state
            self.graph.add_node(
                "start",
                lambda state: {
                    "research_question": state.get("research_question", ""),
                    "planner_response": [],
                    "selector_response": [],
                    "reporter_response": [],
                    "reviewer_response": [],
                    "router_response": [],
                    "serper_response": [],
                    "scraper_response": [],
                    "final_reports": [],
                    "end_chain": [HumanMessage(content="false")],
                },
            )

            # Set entry point first
            self.graph.set_entry_point("start")

            # Create and add nodes
            nodes = self._create_nodes()
            self._add_nodes_to_graph(nodes)

            # Add edges
            self._add_edges()

            logger.info("Successfully built graph")
            return self.graph
        except Exception as e:
            logger.error(f"Failed to build graph: {str(e)}")
            raise Exception(f"Graph building failed: {str(e)}")

    def visualize(self, graph) -> None:
        """
        Visualize the graph structure.
        """
        workflow = graph.compile()
        output_path = "research_agent_graph.png"

        try:
            display(
                Image(
                    workflow.get_graph().draw_mermaid_png(
                        curve_style=CurveStyle.LINEAR,
                        node_colors=NodeStyles(
                            first="#ffdfba", last="#baffc9", default="#fad7de"
                        ),
                        wrap_label_n_words=2,
                        output_file_path=output_path,
                        draw_method=MermaidDrawMethod.PYPPETEER,
                        background_color="white",
                        padding=20,
                    )
                )
            )
        except Exception as e:
            logger.error(f"Failed to visualize graph: {str(e)}")
            # Print more detailed error information
            print("\nGraph Structure:")
            print("Nodes:", list(workflow.get_graph().nodes))
            print(
                "\nRegular Edges:",
                [e for e in workflow.get_graph().edges if not e.conditional],
            )
            print(
                "\nConditional Edges:",
                [e for e in workflow.get_graph().edges if e.conditional],
            )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test graph building
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        server: str = "openai"
        model: str = "gpt-4"
        stop: list = None
        model_endpoint: str = None
        temperature: float = 0.7

    config = TestConfig()
    builder = AgentGraphBuilder(config)
    graph = builder.build()
    builder.visualize(graph)
    logger.info("Successfully created test graph")

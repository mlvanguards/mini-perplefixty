import json
import logging
from typing import Any, Dict, List

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.graph import END, StateGraph
from retrieval.abastract_retrieval.strategies.base import (
    BaseQdrantRetriever,
    RetrievalConfig,
)
from retrieval.core.exceptions import (
    ComponentInitializationError,
    GraphBuildError,
    NodeCreationError,
)
from retrieval.core.interfaces import (
    IMemoryService,
    IRetrieverService,
)
from retrieval.db.base_manager import QdrantConfig
from retrieval.db.memory_manager import MemoryManager
from retrieval.external.tenant_context import get_tenant_context
from retrieval.rag.agentic_rag.agent_graph.nodes.aggregator import AggregatorNode
from retrieval.rag.agentic_rag.agent_graph.nodes.conversational_decision import (
    ConversationalDecisionNode,
)
from retrieval.rag.agentic_rag.agent_graph.nodes.memory import MemoryRetrievalNode
from retrieval.rag.agentic_rag.agent_graph.nodes.query_rewriter import QueryRewriterNode
from retrieval.rag.agentic_rag.agent_graph.nodes.reviewer import ReviewerNode
from retrieval.rag.agentic_rag.agent_graph.nodes.search_documents import (
    SearchDocumentsNode,
)
from retrieval.rag.agentic_rag.graph_config.config import GraphConfig
from retrieval.rag.agentic_rag.services.memory import MemoryService
from retrieval.rag.agentic_rag.services.retriever import (
    RetrieverService,
)
from retrieval.rag.agentic_rag.states.state import AgentGraphState

logger = logging.getLogger(__name__)


class AgentGraphBuilder:
    """
    Builder for creating and configuring the agent graph.

    This class handles the construction of the complete agent graph, including:
    - Component initialization
    - Node creation and configuration
    - Edge definition
    - State management

    Attributes:
        config (GraphConfig): Configuration for the graph
        retriever_service (IRetrieverService): Service for creating retrievers
        memory_service (IMemoryService): Service for managing memory
        graph (StateGraph): The graph being constructed
        retriever (BaseQdrantRetriever): The configured retriever
        memory_manager (MemoryManager): The configured memory manager
    """

    def __init__(
        self,
        config: GraphConfig,
        retriever_service: IRetrieverService = RetrieverService(),
        memory_service: IMemoryService = MemoryService(),
    ):
        """
        Initialize the graph builder.

        Args:
            config: Configuration for the graph
            retriever_service: Service for creating retrievers
            memory_service: Service for managing memory

        Raises:
            ComponentInitializationError: If initialization fails
        """
        self.config = config
        self.retriever_service = retriever_service
        self.memory_service = memory_service

        try:
            self.graph = StateGraph(AgentGraphState)
            self._initialize_components()
        except Exception as e:
            logger.error(f"Failed to initialize graph builder: {str(e)}")
            raise ComponentInitializationError(f"Graph initialization failed: {str(e)}")

    def _initialize_components(self) -> None:
        """
        Initialize all required components and services.

        Raises:
            ComponentInitializationError: If component initialization fails
        """
        try:
            tenant_context = get_tenant_context(self.config.tenant_id)
            # Store collection_name from tenant_context
            self.collection_name = tenant_context["qdrant"]["collection"]
            logger.info(f"Initialized collection name: {self.collection_name}")

            self.retriever = self._create_retriever(tenant_context)
            logger.info(f"Created retriever for collection: {self.collection_name}")

            self.memory_manager = self._create_memory_manager(tenant_context)
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise ComponentInitializationError(
                f"Failed to initialize components: {str(e)}"
            )

    def _create_retriever(self, tenant_context: dict) -> BaseQdrantRetriever:
        """
        Create and configure the retriever.

        Args:
            settings: Application settings

        Returns:
            Configured retriever instance

        Raises:
            ComponentInitializationError: If retriever creation fails
        """
        try:
            retrieval_config = RetrievalConfig(
                qdrant_host=tenant_context["qdrant"]["host"].rstrip("/"),
                qdrant_api_key=tenant_context["qdrant"]["api_key"],
                timeout=60,
            )
            return self.retriever_service.create_retriever(retrieval_config)
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}")
            raise ComponentInitializationError(f"Retriever creation failed: {str(e)}")

    def _create_memory_manager(self, tenant_context: dict) -> MemoryManager:
        """
        Create and configure the memory manager.

        Args:
            settings: Application settings

        Returns:
            Configured memory manager instance

        Raises:
            ComponentInitializationError: If memory manager creation fails
        """
        try:
            memory_config = QdrantConfig(
                url=tenant_context["qdrant"]["host"].rstrip("/"),
                api_key=tenant_context["qdrant"]["api_key"],
                collection_name=f"{self.config.tenant_id}_memory",
            )
            return self.memory_service.create_memory_manager(memory_config)
        except Exception as e:
            logger.error(f"Failed to create memory manager: {str(e)}")
            raise ComponentInitializationError(
                f"Memory manager creation failed: {str(e)}"
            )

    def _create_initial_state(self) -> Dict[str, Any]:
        """
        Create the initial state for the graph.

        Returns:
            Dictionary containing the initial state
        """
        return {
            "tenant_id": self.config.tenant_id,
            "workflow_id": self.config.workflow_id,
            "user_id": self.config.user_id,
            "user_query": None,
            "memory_response": [],
            "query_rewriter_response": [],
            "search_documents_response": [],
            "aggregator_response": [],
            "reviewer_response": [],
            "collection_name": self.collection_name,
        }

    def _create_nodes(self) -> Dict[str, Any]:
        """
        Create all nodes for the graph.

        Returns:
            Dictionary of node instances

        Raises:
            NodeCreationError: If node creation fails
        """
        try:
            return {
                "memory": MemoryRetrievalNode(self.memory_manager),
                "decision": ConversationalDecisionNode(
                    model=self.config.model,
                    server=self.config.server,
                    stop=self.config.stop,
                    model_endpoint=self.config.model_endpoint,
                    temperature=self.config.temperature,
                ),
                "rewriter": QueryRewriterNode(
                    model=self.config.model,
                    server=self.config.server,
                    stop=self.config.stop,
                    model_endpoint=self.config.model_endpoint,
                    temperature=self.config.temperature,
                ),
                "search": SearchDocumentsNode(
                    retriever=self.retriever,
                    ranker=self.config.ranker,
                    collection_name=self.config.collection_name,
                ),
                "aggregator": AggregatorNode(
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
            }
        except Exception as e:
            logger.error(f"Failed to create nodes: {str(e)}")
            raise NodeCreationError(f"Node creation failed: {str(e)}")

    def _add_nodes_to_graph(self, nodes: Dict[str, Any]) -> None:
        """
        Add nodes to the graph.

        Args:
            nodes: Dictionary of nodes to add

        Raises:
            GraphBuildError: If adding nodes fails
        """
        try:
            for node in nodes.values():
                self.graph.add_node(node.name, node.process)
            logger.info(f"Successfully added {len(nodes)} nodes to graph")
        except Exception as e:
            logger.error(f"Failed to add nodes to graph: {str(e)}")
            raise GraphBuildError(f"Failed to add nodes: {str(e)}")

    def _add_edges(self) -> None:
        """
        Add all edges to the graph.

        Raises:
            GraphBuildError: If adding edges fails
        """
        try:
            # Basic edges
            self._add_basic_edges()

            # Conditional edges
            self.graph.add_conditional_edges(
                "conversational_decision", self._route_conversation
            )
            logger.info("Successfully added all edges to graph")
        except Exception as e:
            logger.error(f"Failed to add edges to graph: {str(e)}")
            raise GraphBuildError(f"Failed to add edges: {str(e)}")

    def _add_basic_edges(self) -> None:
        """Add basic edges to the graph."""
        edges: List[tuple] = [
            ("start", "retrieve_memory"),
            ("retrieve_memory", "conversational_decision"),
            ("query_rewriter", "search_documents"),
            ("search_documents", "aggregator"),
            ("aggregator", END),
            # ("reviewer", END),
        ]

        for source, target in edges:
            self.graph.add_edge(source, target)

    @staticmethod
    def _route_conversation(state: Dict[str, Any]) -> str:
        """
        Route the conversation based on the decision response.

        Args:
            state: Current state dictionary

        Returns:
            String indicating the next node
        """
        decision_response = state["conversational_decision"]

        if isinstance(decision_response, str):
            try:
                decision_response = json.loads(decision_response)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse decision response: {str(e)}")
                return "query_rewriter"

        return (
            "query_rewriter"
            if decision_response.get("query_relation") == "new"
            else "aggregator"
        )

    def build(self) -> StateGraph:
        """
        Build and return the configured graph.

        Returns:
            Configured StateGraph instance

        Raises:
            GraphBuildError: If graph building fails
        """
        try:
            # Add start node
            self.graph.add_node(
                "start",
                lambda state: {
                    **self._create_initial_state(),
                    **(state if isinstance(state, dict) else {}),
                },
            )
            self.graph.set_entry_point("start")

            # Create and add nodes
            nodes = self._create_nodes()
            self._add_nodes_to_graph(nodes)

            # Define edges
            self._add_edges()

            logger.info("Successfully built graph")
            return self.graph
        except Exception as e:
            logger.error(f"Failed to build graph: {str(e)}")
            raise GraphBuildError(f"Graph building failed: {str(e)}")

    def visualize(self, graph) -> None:
        workflow = graph.compile()
        output_path = "new_agent_graph.png"

        display(
            Image(
                workflow.get_graph().draw_mermaid_png(
                    curve_style=CurveStyle.LINEAR,
                    node_colors=NodeStyles(
                        first="#ffdfba", last="#baffc9", default="#fad7de"
                    ),
                    wrap_label_n_words=9,
                    output_file_path=output_path,
                    draw_method=MermaidDrawMethod.PYPPETEER,
                    background_color="white",
                    padding=10,
                )
            )
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test graph building
    try:
        config = GraphConfig(
            server="openai",
            model="gpt-4-mini",
            tenant_id="test",
        )
        builder = AgentGraphBuilder(config)
        graph = builder.build()
        builder.visualize(graph)
        logger.info("Successfully created test graph")
    except GraphBuildError as e:
        logger.error(f"Failed to create test graph: {str(e)}")

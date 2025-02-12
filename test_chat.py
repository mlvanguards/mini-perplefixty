from langchain.schema import HumanMessage

from src.builder.config import GraphConfig
from src.builder.graph import AgentGraphBuilder


def main():
    # Configuration
    server = "openai"
    model = "gpt-4o-mini"
    model_endpoint = None
    iterations = 40

    # Create graph config
    config = GraphConfig(
        server=server,
        model=model,
        model_endpoint=model_endpoint,
    )

    # Create graph using the builder
    builder = AgentGraphBuilder(config)
    graph = builder.build()

    # Compile the workflow
    workflow = graph.compile()
    print("Graph and workflow created.")

    # Main interaction loop
    verbose = False
    # query = input("\nPlease enter your research question (or 'quit' to exit): ")
    query = "What is the capital of France?"

    # Prepare inputs for the workflow
    dict_inputs = {
        "research_question": query,
        "planner_response": [],
        "selector_response": [],
        "reporter_response": [],
        "reviewer_response": [],
        "router_response": [],
        "serper_response": [],
        "scraper_response": [],
        "final_reports": [],
        "end_chain": [HumanMessage(content="false")],
    }
    limit = {"recursion_limit": iterations}

    # Stream results
    for event in workflow.stream(dict_inputs, limit):
        if verbose:
            print("\nState Dictionary:", event)
        else:
            print("\n")


if __name__ == "__main__":
    main()

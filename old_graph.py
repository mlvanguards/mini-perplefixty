from langchain_core.messages import HumanMessage

test = {
    "research_question": "What is the capital of France?",
    "planner_response": [
        HumanMessage(
            content='{\n    "search_term": "capital of France",\n    "overall_strategy": "Begin by searching for the capital of France using the search term provided. Look for reliable sources such as government websites, educational institutions, or reputable news outlets to confirm the information.",\n    "additional_information": "If the initial search does not yield satisfactory results, consider using variations such as \'France capital city\' or \'Paris capital of France\'. Additionally, applying filters for recent articles or specific types of sources (like Wikipedia or official tourism sites) may help narrow down the results."\n}',
            additional_kwargs={},
            response_metadata={},
            id="534eff30-d191-48f5-9ee7-edff19341aa0",
        )
    ],
    "selector_response": [],
    "reporter_response": [],
    "reviewer_response": [],
    "router_response": [],
    "serper_response": [],
    "scraper_response": [],
    "final_reports": [],
    "end_chain": [
        HumanMessage(
            content="false",
            additional_kwargs={},
            response_metadata={},
            id="bcf18f7f-1299-4ba1-af4b-4b9748272080",
        ),
        HumanMessage(
            content="false",
            additional_kwargs={},
            response_metadata={},
            id="3130b701-de76-427a-9c6d-347919f1a07c",
        ),
    ],
}

print(test["planner_response"][-1].content)

planner_prompt_template = """
You are a planner. Your responsibility is to create a comprehensive plan to help your team answer a research question. 
Questions may vary from simple to complex, multi-step queries. Your plan should provide appropriate guidance for your 
team to use an internet search engine effectively.

Focus on highlighting the most relevant search term to start with, as another team member will use your suggestions 
to search for relevant information.

If you receive feedback, you must adjust your plan accordingly. Here is the feedback received:
Feedback: {feedback}

Current date and time:
{datetime}

Your response must take the following json format:

    "search_term": "The most relevant search term to start with"
    "overall_strategy": "The overall strategy to guide the search process"
    "additional_information": "Any additional information to guide the search including other search terms or filters"

"""

planner_guided_json = {
    "type": "object",
    "properties": {
        "search_term": {
            "type": "string",
            "description": "The most relevant search term to start with",
        },
        "overall_strategy": {
            "type": "string",
            "description": "The overall strategy to guide the search process",
        },
        "additional_information": {
            "type": "string",
            "description": "Any additional information to guide the search including other search terms or filters",
        },
    },
    "required": ["search_term", "overall_strategy", "additional_information"],
}

selector_prompt_template = """
You are a selector. You will be presented with a search engine results page containing a list of potentially relevant 
search results. Your task is to read through these results, select the most relevant one, and provide a comprehensive 
reason for your selection.

here is the search engine results page:
{serp}

Return your findings in the following json format:

    "selected_page_url": "The exact URL of the page you selected",
    "description": "A brief description of the page",
    "reason_for_selection": "Why you selected this page"


Adjust your selection based on any feedback received:
Feedback: {feedback}

Here are your previous selections:
{previous_selections}
Consider this information when making your new selection.

Current date and time:
{datetime}
"""

selector_guided_json = {
    "type": "object",
    "properties": {
        "selected_page_url": {
            "type": "string",
            "description": "The exact URL of the page you selected",
        },
        "description": {
            "type": "string",
            "description": "A brief description of the page",
        },
        "reason_for_selection": {
            "type": "string",
            "description": "Why you selected this page",
        },
    },
    "required": ["selected_page_url", "description", "reason_for_selection"],
}

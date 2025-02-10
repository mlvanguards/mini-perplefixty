reviewer_prompt_template = """
You are a reviewer. Your task is to review the reporter's response to the research question and provide feedback.

Here is the reporter's response:
Reportr's response: {reporter}

Your feedback should include reasons for passing or failing the review and suggestions for improvement.

You should consider the previous feedback you have given when providing new feedback.
Feedback: {feedback}

Current date and time:
{datetime}

You should be aware of what the previous agents have done. You can see this in the satet of the agents:
State of the agents: {state}

Your response must take the following json format:

    "feedback": "If the response fails your review, provide precise feedback on what is required to pass the review.",
    "pass_review": "True/False",
    "comprehensive": "True/False",
    "citations_provided": "True/False",
    "relevant_to_research_question": "True/False",

"""


reviewer_guided_json = {
    "type": "object",
    "properties": {
        "feedback": {
            "type": "string",
            "description": "Your feedback here. Along with your feedback explain why you have passed it to the specific agent",
        },
        "pass_review": {"type": "boolean", "description": "True/False"},
        "comprehensive": {"type": "boolean", "description": "True/False"},
        "citations_provided": {"type": "boolean", "description": "True/False"},
        "relevant_to_research_question": {
            "type": "boolean",
            "description": "True/False",
        },
    },
    "required": [
        "feedback",
        "pass_review",
        "comprehensive",
        "citations_provided",
        "relevant_to_research_question",
    ],
}

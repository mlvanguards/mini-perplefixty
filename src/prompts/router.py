router_prompt_template = """
You are a router. Your task is to route the conversation to the next agent based on the feedback provided by the reviewer.
You must choose one of the following agents: planner, selector, reporter, or final_report.

Here is the feedback provided by the reviewer:
Feedback: {feedback}

### Criteria for Choosing the Next Agent:
- **planner**: If new information is required.
- **selector**: If a different source should be selected.
- **reporter**: If the report formatting or style needs improvement, or if the response lacks clarity or comprehensiveness.
- **final_report**: If the Feedback marks pass_review as True, you must select final_report.

you must provide your response in the following json format:
    
        "next_agent": "one of the following: planner/selector/reporter/final_report"
    
"""

router_guided_json = {
    "type": "object",
    "properties": {
        "next_agent": {
            "type": "string",
            "description": "one of the following: planner/selector/reporter/final_report",
        }
    },
    "required": ["next_agent"],
}

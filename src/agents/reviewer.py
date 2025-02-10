from termcolor import colored

from src.agents.base import Agent
from src.prompts.reviewer import reviewer_prompt_template
from src.utils.helper_functions import check_for_content, get_current_utc_datetime


class ReviewerAgent(Agent):
    def invoke(
        self,
        research_question,
        prompt=reviewer_prompt_template,
        reporter=None,
        feedback=None,
    ):
        reporter_value = reporter() if callable(reporter) else reporter
        feedback_value = feedback() if callable(feedback) else feedback

        reporter_value = check_for_content(reporter_value)
        feedback_value = check_for_content(feedback_value)

        reviewer_prompt = prompt.format(
            reporter=reporter_value,
            state=self.state,
            feedback=feedback_value,
            datetime=get_current_utc_datetime(),
        )

        messages = [
            {"role": "system", "content": reviewer_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"Reviewer 👩🏽‍⚖️: {response}", "magenta"))
        self.update_state("reviewer_response", response)
        return self.state

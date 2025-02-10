from termcolor import colored

from src.agents.base import Agent
from src.prompts.planner import planner_prompt_template
from src.utils.helper_functions import check_for_content, get_current_utc_datetime


class PlannerAgent(Agent):
    def invoke(self, research_question, prompt=planner_prompt_template, feedback=None):
        feedback_value = feedback() if callable(feedback) else feedback
        feedback_value = check_for_content(feedback_value)

        planner_prompt = prompt.format(
            feedback=feedback_value, datetime=get_current_utc_datetime()
        )

        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        self.update_state("planner_response", response)
        print(colored(f"Planner 👩🏿‍💻: {response}", "cyan"))
        return self.state

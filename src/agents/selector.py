from termcolor import colored

from src.agents.base import Agent
from src.prompts.selector import selector_prompt_template
from src.utils.helper_functions import check_for_content, get_current_utc_datetime


class SelectorAgent(Agent):
    def invoke(
        self,
        research_question,
        prompt=selector_prompt_template,
        feedback=None,
        previous_selections=None,
        serp=None,
    ):
        feedback_value = feedback() if callable(feedback) else feedback
        previous_selections_value = (
            previous_selections()
            if callable(previous_selections)
            else previous_selections
        )

        feedback_value = check_for_content(feedback_value)
        previous_selections_value = check_for_content(previous_selections_value)

        selector_prompt = prompt.format(
            feedback=feedback_value,
            previous_selections=previous_selections_value,
            serp=serp().content,
            datetime=get_current_utc_datetime(),
        )

        messages = [
            {"role": "system", "content": selector_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"selector 🧑🏼‍💻: {response}", "green"))
        self.update_state("selector_response", response)
        return self.state

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
        # Safely handle potentially callable inputs
        try:
            feedback_value = feedback() if callable(feedback) else feedback
        except (TypeError, AttributeError):
            feedback_value = None

        try:
            previous_selections_value = (
                previous_selections()
                if callable(previous_selections)
                else previous_selections
            )
        except (TypeError, AttributeError):
            previous_selections_value = None

        try:
            serp_content = (
                serp().content if callable(serp) else serp.content if serp else ""
            )
        except (TypeError, AttributeError):
            serp_content = ""

        # Apply content checks
        feedback_value = check_for_content(feedback_value)
        previous_selections_value = check_for_content(previous_selections_value)

        # Format prompt with safe values
        selector_prompt = prompt.format(
            feedback=feedback_value,
            previous_selections=previous_selections_value,
            serp=serp_content,
            datetime=get_current_utc_datetime(),
        )

        # Create messages for LLM
        messages = [
            {"role": "system", "content": selector_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        # Get LLM response
        try:
            llm = self.get_llm()
            ai_msg = llm.invoke(messages)
            response = ai_msg.content
            print(colored(f"Selector 🧑🏼‍💻: {response}", "green"))
            self.update_state("selector_response", response)
        except Exception as e:
            print(colored(f"Error in selector processing: {str(e)}", "red"))
            response = ""

        return self.state

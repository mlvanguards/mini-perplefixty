from termcolor import colored

from src.agents.base import Agent
from src.prompts.router import router_prompt_template
from src.utils.helper_functions import check_for_content


class RouterAgent(Agent):
    def invoke(
        self, feedback=None, research_question=None, prompt=router_prompt_template
    ):
        feedback_value = feedback() if callable(feedback) else feedback
        feedback_value = check_for_content(feedback_value)

        router_prompt = prompt.format(feedback=feedback_value)

        messages = [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        llm = self.get_llm()
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"Router 🧭: {response}", "blue"))
        self.update_state("router_response", response)
        return self.state

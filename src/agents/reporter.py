from termcolor import colored

from src.agents.base import Agent
from src.prompts.reporter import reporter_prompt_template
from src.utils.helper_functions import check_for_content, get_current_utc_datetime


class ReporterAgent(Agent):
    def invoke(
        self,
        research_question,
        prompt=reporter_prompt_template,
        feedback=None,
        previous_reports=None,
        research=None,
    ):
        feedback_value = feedback() if callable(feedback) else feedback
        previous_reports_value = (
            previous_reports() if callable(previous_reports) else previous_reports
        )
        research_value = research() if callable(research) else research

        feedback_value = check_for_content(feedback_value)
        previous_reports_value = check_for_content(previous_reports_value)
        research_value = check_for_content(research_value)

        reporter_prompt = prompt.format(
            feedback=feedback_value,
            previous_reports=previous_reports_value,
            datetime=get_current_utc_datetime(),
            research=research_value,
        )

        messages = [
            {"role": "system", "content": reporter_prompt},
            {"role": "user", "content": f"research question: {research_question}"},
        ]

        llm = self.get_llm(json_model=False)
        ai_msg = llm.invoke(messages)
        response = ai_msg.content

        print(colored(f"Reporter 👨‍💻: {response}", "yellow"))
        self.update_state("reporter_response", response)
        return self.state

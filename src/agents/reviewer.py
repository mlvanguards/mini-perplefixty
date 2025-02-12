from termcolor import colored

from src.agents.base import Agent
from src.prompts.reviewer import reviewer_prompt_template
from src.utils.helper_functions import get_current_utc_datetime


class ReviewerAgent(Agent):
    def invoke(
        self,
        state_input: dict,
        prompt=reviewer_prompt_template,
        feedback=None,
        state=None,
    ):
        """
        Invoke the reviewer agent with input state.

        Args:
            state_input (dict): Input state containing research_question and report_content
            prompt: The prompt template to use
        """
        try:
            # Extract input values
            input_data = state_input.get("input", {})
            research_question = input_data.get("research_question", "")
            report_content = input_data.get("report_content", "")

            print(
                colored(f"Reviewing report for question: {research_question}", "cyan")
            )

            # Format the prompt with the report content
            reviewer_prompt = prompt.format(
                report_content=report_content,  # Changed from reporter to report_content
                datetime=get_current_utc_datetime(),
                feedback=feedback,
                state=state,
            )

            # Create messages for LLM
            messages = [
                {"role": "system", "content": reviewer_prompt},
                {"role": "user", "content": f"research question: {research_question}"},
            ]

            # Get LLM response
            llm = self.get_llm()
            ai_msg = llm.invoke(messages)
            response = ai_msg.content

            # Update state with the response
            self.state = {"reviewer_response": response}

            print(colored(f"Reviewer 👩🏽‍⚖️: {response}", "magenta"))

            return self.state

        except Exception as e:
            print(colored(f"Error in ReviewerAgent: {str(e)}", "red"))
            self.state = {"reviewer_response": f"Error in review: {str(e)}"}
            return self.state

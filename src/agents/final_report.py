from termcolor import colored

from src.agents.base import Agent


class FinalReportAgent(Agent):
    def invoke(self, final_response=None):
        final_response_value = (
            final_response() if callable(final_response) else final_response
        )
        response = final_response_value.content

        print(colored(f"Final Report 📝: {response}", "blue"))
        self.update_state("final_reports", response)
        return self.state

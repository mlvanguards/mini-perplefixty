import json

from termcolor import colored

from src.agents.base import Agent
from src.prompts.router import router_guided_json, router_prompt_template


class RouterAgent(Agent):
    def invoke(self, state_input: dict, prompt=router_prompt_template):
        """
        Invoke the router agent with input state.

        Args:
            state_input (dict): Input state containing research_question and current_state
            prompt: The prompt template to use
        """
        try:
            input_data = state_input.get("input", {})
            research_question = input_data.get("research_question", "")
            current_state = input_data.get("current_state", {})

            # Get the reviewer feedback from the state
            reviewer_messages = current_state.get("reviewer_response", [])
            if not reviewer_messages:
                feedback = "No reviewer feedback available"
            else:
                last_review = reviewer_messages[-1]
                try:
                    # Parse the nested structure
                    review_data = json.loads(last_review.content)
                    reviewer_response = json.loads(
                        review_data["content"]["reviewer_response"]
                    )
                    feedback = json.dumps(reviewer_response, indent=2)
                except json.JSONDecodeError:
                    feedback = str(last_review.content)

            # Format prompt
            router_prompt = prompt.format(feedback=feedback)

            # Create messages for LLM
            messages = [
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": f"research question: {research_question}"},
                {
                    "role": "system",
                    "content": f"Please provide your response in the following JSON structure: {router_guided_json}",
                },
            ]

            # Get LLM response
            llm = self.get_llm()
            ai_msg = llm.invoke(messages)
            response = ai_msg.content

            print(colored(f"Router 🧭: {response}", "blue"))

            # Update state
            self.state = {"router_response": response}
            return self.state

        except Exception as e:
            print(colored(f"Error in RouterAgent: {str(e)}", "red"))
            self.state = {
                "router_response": json.dumps(
                    {
                        "next_agent": "final_report",  # Default to final_report on error
                    }
                )
            }
            return self.state

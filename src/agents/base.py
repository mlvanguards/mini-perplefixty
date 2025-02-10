from src.models.openai_models import get_open_ai, get_open_ai_json
from src.states.state import AgentGraphState


class Agent:
    def __init__(
        self,
        state: AgentGraphState,
        model=None,
        server=None,
        temperature=0,
        model_endpoint=None,
        stop=None,
        guided_json=None,
    ):
        self.state = state
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.guided_json = guided_json

    def get_llm(self, json_model=True):
        if self.server == "openai":
            return (
                get_open_ai_json(model=self.model, temperature=self.temperature)
                if json_model
                else get_open_ai(model=self.model, temperature=self.temperature)
            )

    def update_state(self, key, value):
        self.state = {**self.state, key: value}

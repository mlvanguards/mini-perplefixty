from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Add the missing fields
    OPENAI_API_KEY: str
    SERPER_API_KEY: str

    PYTHONPATH: str

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()

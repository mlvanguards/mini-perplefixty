from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphConfig:
    """Configuration for the agent graph."""

    server: Optional[str] = None
    model: Optional[str] = None
    stop: Optional[list] = None
    model_endpoint: Optional[str] = None
    temperature: float = 0

from abc import ABC, abstractmethod
from typing import Any, Dict


class GraphNode(ABC):
    """Base abstract class for all graph nodes."""

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updated state."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the node name."""
        pass

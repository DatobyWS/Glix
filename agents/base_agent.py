import abc
import logging
import time
from typing import Any, Dict


class BaseAgent(abc.ABC):
    """Abstract base class for all pipeline agents."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"glix.{name}")

    @abc.abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            context: Pipeline state dict passed between agents.
                     Each agent reads what it needs and adds its results.
        Returns:
            Updated context dict with this agent's output added.
        """
        pass

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Wraps run() with logging and timing."""
        self.logger.info(f"[{self.name}] Starting...")
        start = time.time()
        try:
            result = self.run(context)
            elapsed = time.time() - start
            self.logger.info(f"[{self.name}] Completed in {elapsed:.1f}s")
            return result
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed: {e}", exc_info=True)
            raise

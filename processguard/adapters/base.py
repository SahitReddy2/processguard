from __future__ import annotations
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Adapter base: normalizes framework-specific events into AgentEvents."""

    @abstractmethod
    def attach(self, framework_object):
        """Hook into the framework object and start emitting events."""
        ...

    def detach(self):
        """Remove hooks (optional)."""
        pass

"""Base class and shared helpers for dashboard builders."""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict


class DashboardBuilderBase(ABC):
    """Abstract base for building result -> HTML dashboards."""

    @abstractmethod
    def build_dashboard(self, result: Any) -> str:
        """Build a complete HTML dashboard from the given result. Must be implemented by subclasses."""
        ...


def serialize_payload_for_js(data: Dict[str, Any]) -> str:
    """
    Serialize a payload dict to JSON string for embedding in HTML/JS.
    Handles numpy arrays and other non-JSON-serializable types via a default converter.
    """
    def _default(o: Any) -> Any:
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(data, indent=2, default=_default)


def section_wrapper(inner_html: str, title: str, section_id: str = "") -> str:
    """Wrap HTML content in a titled section block."""
    id_attr = f' id="{section_id}"' if section_id else ""
    return f'<div class="plot-container"{id_attr}><h2>{title}</h2>{inner_html}</div>'

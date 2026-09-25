"""One description of every provider, for both the factory and the UI.

The frontend used to hard-code its provider list, so it silently fell two
providers behind the backend: ``servus`` and ``agentic`` were registered in
``get_provider`` but unreachable from the app. Anything that needs to know what
providers exist reads this module instead.

Adding a provider means adding one entry here and one branch in
``providers.get_provider``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# How the UI should let someone pick a model for a provider.
INPUT_LIST = "list"    # choose from the static `models` list
INPUT_FETCH = "fetch"  # fetch the installed list from an endpoint
INPUT_PATH = "path"    # free text: a filesystem path or Hugging Face id
INPUT_TEXT = "text"    # free text: an opaque identifier


@dataclass
class ProviderInfo:
    name: str
    label: str
    kind: str                                  # api | local | service
    model_input: str
    description: str
    models: List[str] = field(default_factory=list)
    requires_api_key: Optional[str] = None     # settings field that must be set
    fetch_endpoint: Optional[str] = None
    placeholder: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    requires_extra: Optional[str] = None       # optional pip extra

    def to_dict(self) -> Dict:
        return asdict(self)


PROVIDERS: List[ProviderInfo] = [
    ProviderInfo(
        name="openai",
        label="OpenAI",
        kind="api",
        model_input=INPUT_TEXT,
        description="Enter a Chat Completions model ID available to your OpenAI account.",
        placeholder="Model ID from your provider account",
        requires_api_key="openai_api_key",
    ),
    ProviderInfo(
        name="anthropic",
        label="Anthropic",
        kind="api",
        model_input=INPUT_TEXT,
        description="Enter a Claude Messages model ID available to your Anthropic account. Generation seeds are unavailable.",
        placeholder="Model ID from your provider account",
        requires_api_key="anthropic_api_key",
    ),
    ProviderInfo(
        name="google",
        label="Google",
        kind="api",
        model_input=INPUT_TEXT,
        description="Enter a Gemini model ID available to your Google account. This integration does not support generation seeds.",
        placeholder="Model ID from your provider account",
        requires_api_key="google_api_key",
    ),
    ProviderInfo(
        name="ollama",
        label="Ollama (local)",
        kind="local",
        model_input=INPUT_FETCH,
        description="Models installed in a local Ollama instance.",
        fetch_endpoint="/api/v1/models/ollama",
    ),
    ProviderInfo(
        name="transformers",
        label="Hugging Face / server models",
        kind="local",
        model_input=INPUT_PATH,
        description=(
            "Run a public Hugging Face model or one of your custom checkpoints "
            "on the connected analysis server. Public models need no sign-in; "
            "gated and private models need an authorized server token."
        ),
        placeholder="models/ablated  or  Qwen/Qwen2.5-3B-Instruct",
        aliases=["local"],
        requires_extra="interp",
    ),
    ProviderInfo(
        name="servus",
        label="Servus (VivaOS)",
        kind="service",
        model_input=INPUT_TEXT,
        description=(
            "Drives a model through the servus gate, so responses carry cognitiond "
            "catch metadata. The model name is passed through to servus."
        ),
        placeholder="model name passed to servus",
    ),
    ProviderInfo(
        name="agentic_a2a",
        label="Agentic A2A team",
        kind="service",
        model_input=INPUT_TEXT,
        description=(
            "Drives a deterministic LangGraph team over A2A. The model name selects "
            "the team (e.g. static_team)."
        ),
        placeholder="static_team",
        aliases=["agentic"],
    ),
]

_BY_NAME: Dict[str, ProviderInfo] = {}
for _info in PROVIDERS:
    _BY_NAME[_info.name] = _info
    for _alias in _info.aliases:
        _BY_NAME[_alias] = _info


def get_provider_info(name: str) -> Optional[ProviderInfo]:
    """Look up a provider's metadata by canonical name or alias."""
    return _BY_NAME.get((name or "").lower())


def list_providers(include_aliases: bool = False) -> List[ProviderInfo]:
    return list(_BY_NAME.values()) if include_aliases else list(PROVIDERS)


def provider_names(include_aliases: bool = True) -> List[str]:
    return sorted(_BY_NAME) if include_aliases else [p.name for p in PROVIDERS]

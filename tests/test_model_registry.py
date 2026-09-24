"""The provider registry must stay in step with the factory.

The frontend previously hard-coded its provider list and fell two providers
behind the backend. These tests fail if the registry and `get_provider` drift
apart in either direction.
"""

import pytest

from vivasecuris.aiasylum.exceptions import ModelProviderError
from vivasecuris.aiasylum.models.registry import (
    get_provider_info,
    list_providers,
    provider_names,
)


def test_every_registry_entry_is_constructible():
    """A provider listed to the UI must be one the factory can actually build."""
    from vivasecuris.aiasylum.models.providers import get_provider

    for name in provider_names():
        try:
            get_provider(name)
        except ModelProviderError as exc:
            # Missing credentials are fine; an unknown provider is not.
            assert "Unknown provider" not in str(exc), f"{name} is listed but unknown to the factory"


def test_every_factory_provider_is_in_the_registry():
    """The direction that actually broke: registered in code, invisible in the UI."""
    import inspect

    from vivasecuris.aiasylum.models import providers as providers_module

    source = inspect.getsource(providers_module.get_provider)
    keys = {
        line.split('"')[1]
        for line in source.splitlines()
        if line.strip().startswith('"') and ":" in line
    }
    missing = {k for k in keys if get_provider_info(k) is None}
    assert not missing, f"providers reachable from get_provider but absent from the registry: {missing}"


def test_aliases_resolve_to_canonical_entries():
    assert get_provider_info("agentic").name == "agentic_a2a"
    assert get_provider_info("local").name == "transformers"
    assert get_provider_info("OLLAMA").name == "ollama"
    assert get_provider_info("nonsense") is None


def test_model_input_mode_is_valid():
    valid = {"list", "fetch", "path", "text"}
    for info in list_providers():
        assert info.model_input in valid, f"{info.name} has an unknown model_input"
        if info.model_input == "list":
            assert info.models, f"{info.name} says 'list' but offers no models"
        if info.model_input == "fetch":
            assert info.fetch_endpoint, f"{info.name} says 'fetch' but has no endpoint"


def test_local_weights_provider_is_exposed():
    """The regression this was written for."""
    info = get_provider_info("transformers")
    assert info is not None
    assert info.kind == "local"
    assert info.model_input == "path"
    assert info.requires_extra == "interp"

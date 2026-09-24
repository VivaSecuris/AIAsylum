"""Ollama-specific CLI commands."""

import asyncio
import click
import httpx

from vivasecuris.aiasylum.models.ollama import OllamaProvider


@click.group()
def ollama():
    """Ollama model management commands."""
    pass


@ollama.command("list")
def list_models():
    """List all available Ollama models."""
    provider = OllamaProvider()

    async def list_available():
        return await provider.list_available_models()

    # list_available_models raises on connection errors (the API relies on that for its 502)
    try:
        models = asyncio.run(list_available())
    except httpx.HTTPError as e:
        raise click.ClickException(
            f"Could not reach Ollama at {provider.base_url}: {e}. Is it running? Start it with: ollama serve"
        )

    if models:
        click.echo("Available Ollama models:")
        for model in models:
            click.echo(f"  - {model}")
    else:
        click.echo("No models found. Make sure Ollama is running: ollama serve")
        click.echo("Pull a model: ollama pull llama2")


@ollama.command()
@click.argument("model_name")
def pull(model_name):
    """Pull an Ollama model."""
    click.echo(f"Pulling model: {model_name}...")

    async def pull_model():
        provider = OllamaProvider()
        return await provider.pull_model(model_name)

    result = asyncio.run(pull_model())

    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
    else:
        click.echo(f"Successfully pulled: {model_name}")


@ollama.command()
@click.argument("model_name")
def info(model_name):
    """Get information about an Ollama model."""
    from vivasecuris.aiasylum.models.ollama import OllamaModel

    async def get_info():
        model = OllamaModel(model_name)
        return await model.get_model_info()

    model_info = asyncio.run(get_info())

    if "error" in model_info:
        click.echo(f"Error: {model_info['error']}", err=True)
    else:
        click.echo(f"Model: {model_name}")
        click.echo(f"Details: {model_info}")


@ollama.command()
@click.argument("model_name")
def check(model_name):
    """Check if an Ollama model is available."""
    from vivasecuris.aiasylum.models.ollama import OllamaModel

    async def check_available():
        model = OllamaModel(model_name)
        return await model.check_available()

    available = asyncio.run(check_available())

    if available:
        click.echo(f"✓ Model '{model_name}' is available")
    else:
        click.echo(f"✗ Model '{model_name}' is not available")
        click.echo(f"Pull it with: ollama pull {model_name}")

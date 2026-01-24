"""Ollama-specific CLI commands."""

import asyncio
import click

from vivasecuris.aiasylum.models.ollama import OllamaProvider


@click.group()
def ollama():
    """Ollama model management commands."""
    pass


@ollama.command()
async def list():
    """List all available Ollama models."""
    provider = OllamaProvider()
    models = await provider.list_available_models()
    
    if models:
        click.echo("Available Ollama models:")
        for model in models:
            click.echo(f"  - {model}")
    else:
        click.echo("No models found. Make sure Ollama is running: ollama serve")
        click.echo("Pull a model: ollama pull llama2")


@ollama.command()
@click.argument("model_name")
async def pull(model_name):
    """Pull an Ollama model."""
    click.echo(f"Pulling model: {model_name}...")
    provider = OllamaProvider()
    result = await provider.pull_model(model_name)
    
    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
    else:
        click.echo(f"Successfully pulled: {model_name}")


@ollama.command()
@click.argument("model_name")
async def info(model_name):
    """Get information about an Ollama model."""
    from vivasecuris.aiasylum.models.ollama import OllamaModel
    
    model = OllamaModel(model_name)
    info = await model.get_model_info()
    
    if "error" in info:
        click.echo(f"Error: {info['error']}", err=True)
    else:
        click.echo(f"Model: {model_name}")
        click.echo(f"Details: {info}")


@ollama.command()
@click.argument("model_name")
async def check(model_name):
    """Check if an Ollama model is available."""
    from vivasecuris.aiasylum.models.ollama import OllamaModel
    
    model = OllamaModel(model_name)
    available = await model.check_available()
    
    if available:
        click.echo(f"✓ Model '{model_name}' is available")
    else:
        click.echo(f"✗ Model '{model_name}' is not available")
        click.echo(f"Pull it with: ollama pull {model_name}")


# Make commands work with async
def _run_async(coro):
    """Run async command."""
    return asyncio.run(coro)


# Wrap async commands
list = click.command()(_run_async(list.__wrapped__))
pull = click.command()(_run_async(pull.__wrapped__))
info = click.command()(_run_async(info.__wrapped__))
check = click.command()(_run_async(check.__wrapped__))

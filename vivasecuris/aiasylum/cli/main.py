"""CLI main entry point."""

import asyncio
import sys
from typing import Optional

import click

from vivasecuris.aiasylum.runner import TestRunner
from vivasecuris.aiasylum.database import get_session, TestRun
from vivasecuris.aiasylum.cli import ollama as ollama_cli


@click.group()
def cli():
    """AI Asylum CLI - LLM Psychoanalysis Framework"""
    pass


@cli.command()
@click.option("--doctor-provider", required=True, help="Doctor model provider")
@click.option("--doctor-model", required=True, help="Doctor model name")
@click.option("--patient-provider", required=True, help="Patient model provider")
@click.option("--patient-model", required=True, help="Patient model name")
@click.option("--test-type", required=True, type=click.Choice(["conversation", "scenario", "adversarial"]))
def run(doctor_provider, doctor_model, patient_provider, patient_model, test_type):
    """Run a test."""
    click.echo(f"Running {test_type} test...")
    click.echo(f"Doctor: {doctor_model} ({doctor_provider})")
    click.echo(f"Patient: {patient_model} ({patient_provider})")
    
    async def run_test():
        runner = TestRunner()
        test_run = await runner.run_test(
            doctor_provider=doctor_provider,
            doctor_model=doctor_model,
            patient_provider=patient_provider,
            patient_model=patient_model,
            test_type=test_type,
        )
        click.echo(f"Test run completed: ID={test_run.id}, Status={test_run.status}")
    
    asyncio.run(run_test())


@cli.command()
@click.option("--limit", default=10, help="Number of results to show")
@click.option("--test-type", help="Filter by test type")
def results(limit, test_type):
    """List test results."""
    runner = TestRunner()
    test_runs = runner.list_test_runs(limit=limit, test_type=test_type)
    
    click.echo(f"\nFound {len(test_runs)} test runs:\n")
    for tr in test_runs:
        click.echo(f"ID: {tr.id} | Type: {tr.test_type} | Status: {tr.status}")
        click.echo(f"  Doctor: {tr.doctor_model} ({tr.doctor_provider})")
        click.echo(f"  Patient: {tr.patient_model} ({tr.patient_provider})")
        click.echo(f"  Created: {tr.created_at}\n")


@cli.command()
def list_benchmarks():
    """List available benchmarks grouped by category."""
    categories = [
        {
            "title": "Knowledge & Reasoning",
            "benchmarks": [
                ("mmlu", "MMLU", "57-subject multiple-choice exam spanning STEM, humanities, law, and medicine"),
                ("arc", "ARC", "Grade-school science questions requiring genuine reasoning"),
                ("math", "MATH", "Competition-level mathematics problems"),
                ("gsm8k", "GSM8K", "Grade-school math word problems requiring multi-step arithmetic"),
            ],
        },
        {
            "title": "Commonsense Reasoning",
            "benchmarks": [
                ("hellaswag", "HellaSwag", "Commonsense NLI — pick the most plausible sentence continuation"),
                ("winogrande", "WinoGrande", "Winograd-schema pronoun resolution for fine-grained commonsense reasoning"),
                ("piqa", "PIQA", "Physical intuition QA — everyday physical interactions"),
            ],
        },
        {
            "title": "Safety & Alignment",
            "benchmarks": [
                ("truthfulqa", "TruthfulQA", "Tests whether the model produces truthful answers instead of plausible-sounding falsehoods"),
                ("bbq", "BBQ", "Reveals social biases across nine protected categories"),
                ("realtoxicityprompts", "RealToxicityPrompts", "Measures how often the model generates toxic continuations"),
                ("jailbreak", "Jailbreak Resistance", "Tests resistance to real-world prompt-injection and jailbreak attacks"),
            ],
        },
    ]

    click.echo("Available benchmarks:\n")
    for category in categories:
        click.echo(f"  {category['title']}")
        click.echo(f"  {'─' * len(category['title'])}")
        for name, title, description in category["benchmarks"]:
            click.echo(f"    {name:25s}  {title} — {description}")
        click.echo()


# Add Ollama subcommands
cli.add_command(ollama_cli.ollama)


@cli.command()
@click.option("--provider", required=True, help="Model provider")
@click.option("--model", required=True, help="Model name")
@click.option("--benchmark", required=True, help="Benchmark name")
@click.option("--num-samples", type=int, help="Number of samples")
def run_benchmark(provider, model, benchmark, num_samples):
    """Run a benchmark."""
    click.echo(f"Running benchmark {benchmark} on {model} ({provider})...")
    
    async def run_benchmark_async():
        from vivasecuris.aiasylum.runner import TestRunner
        
        from vivasecuris.aiasylum.constants import TEST_TYPE_BENCHMARK
        
        runner = TestRunner()
        test_run = await runner.run_test(
            doctor_provider=provider,
            doctor_model=model,
            patient_provider=provider,
            patient_model=model,
            test_type=TEST_TYPE_BENCHMARK,
            test_config={
                "benchmark_name": benchmark,
                "num_samples": num_samples or 100,
                # Jailbreak benchmarks default to one_shot mode
                # Individual prompts will be handled based on their is_multi_shot flag
                "test_mode": "one_shot",
            },
        )
        click.echo(f"Benchmark test run created: ID={test_run.id}, Status={test_run.status}")
        click.echo("Note: Benchmarks run in the background. Check results via API or web UI.")
    
    asyncio.run(run_benchmark_async())


if __name__ == "__main__":
    cli()

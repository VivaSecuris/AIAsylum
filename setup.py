"""Setup script for AI Asylum."""

from setuptools import setup, find_packages

setup(
    name="ai-asylum",
    version="0.1.0",
    description="LLM Psychoanalysis Framework",
    author="VivaSecuris",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "sqlalchemy>=2.0.23",
        "alembic>=1.12.1",
        "openai>=1.3.5",
        "anthropic>=0.7.7",
        "google-generativeai>=0.3.1",
        "httpx>=0.25.1",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "aiasylum=vivasecuris.aiasylum.cli.main:cli",
        ],
    },
)

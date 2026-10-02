"""Setup script for AI Asylum."""

from setuptools import setup, find_namespace_packages

setup(
    name="ai-asylum",
    version="0.1.0",
    description="LLM Psychoanalysis Framework",
    author="VivaSecuris Syndicate",
    license="Proprietary",
    packages=find_namespace_packages(include=["vivasecuris*", "config*"]),
    # pinned pydantic-core has no wheels for 3.13+ (see docs/INSTALL_TROUBLESHOOTING.md)
    python_requires=">=3.10,<3.13",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "sqlalchemy>=2.0.23",
        "alembic>=1.12.1",
        "openai>=1.3.5",
        "anthropic>=0.125.0,<1",
        "google-generativeai>=0.3.1",
        "httpx>=0.25.1",
        "click>=8.1.0",
    ],
    extras_require={
        # Local-weights work: activation capture, interpretability analyses and
        # weight surgery. Kept optional so the default install stays torch-free
        # and the API/CLI continue to install in seconds.
        "interp": [
            "torch>=2.2",
            "transformers>=4.51,<5",
            "huggingface-hub<1.0",
            "accelerate>=0.30",
            "safetensors>=0.4",
            "numpy>=1.24",
            "scikit-learn>=1.3",
            "umap-learn>=0.5",
            # Serves plotly.js from the installed package rather than a CDN.
            # Pinned below 6: plotly.py 6.x bundles plotly.js 3.x, while the
            # vendored dashboard builders were written against plotly.js 2.26.
            "plotly>=5.18,<6",
        ],
        # Gradient-based behavior modification. Plain LoRA only: bitsandbytes
        # has no MPS backend, so 4-bit QLoRA is unavailable on Apple silicon.
        "lora": [
            "peft>=0.13,<1",
            "datasets>=2.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "aiasylum=vivasecuris.aiasylum.cli.main:cli",
        ],
    },
)

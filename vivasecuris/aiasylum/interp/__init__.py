"""Mechanistic interpretability engine.

Vendored from the labotomy project (``llm_prompt_diff``), trimmed to the
analysis core: its FastAPI server, embedded web UI and CLI were dropped because
AI Asylum supplies all three. Model loading was rewritten (see
``core/loader.py``) to remove an Ollama-name mapping that silently substituted
a different model.

Requires the optional ``interp`` extra. Nothing here is imported at package
import time, so a default install stays torch-free.
"""

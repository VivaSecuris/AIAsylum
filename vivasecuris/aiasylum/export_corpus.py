"""Export jailbreak corpus as JSONL for agentic / external harnesses.

No dependency on agentic — writes technique/prompt/source rows that
``AGENTIC_ASYLUM_CORPUS`` consumers can load.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def _load_adversarial_class():
    """Load AdversarialTest without importing tests/__init__ (heavy provider chain)."""
    root = Path(__file__).resolve().parent
    adv_path = root / "tests" / "adversarial.py"
    base_path = root / "tests" / "base.py"

    # Stub modules that adversarial.py imports at top level but are not needed
    # to read the technique catalogs.
    if "vivasecuris.aiasylum.patient" not in sys.modules:
        patient_mod = types.ModuleType("vivasecuris.aiasylum.patient")
        patient_mod.Patient = object  # type: ignore[attr-defined]
        sys.modules["vivasecuris.aiasylum.patient"] = patient_mod

    if "vivasecuris.aiasylum.tests.jailbreak_loader" not in sys.modules:
        loader_mod = types.ModuleType("vivasecuris.aiasylum.tests.jailbreak_loader")
        loader_mod.load_jailbreak_prompts = lambda **kwargs: []  # type: ignore[attr-defined]
        sys.modules["vivasecuris.aiasylum.tests.jailbreak_loader"] = loader_mod

    if "vivasecuris.aiasylum.tests.base" not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(
            "vivasecuris.aiasylum.tests.base", base_path
        )
        if base_spec is None or base_spec.loader is None:
            raise ImportError(f"cannot load {base_path}")
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules["vivasecuris.aiasylum.tests.base"] = base_mod
        base_spec.loader.exec_module(base_mod)

    # Ensure parent packages exist without executing tests/__init__.py
    for name in (
        "vivasecuris",
        "vivasecuris.aiasylum",
        "vivasecuris.aiasylum.tests",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    spec = importlib.util.spec_from_file_location(
        "vivasecuris.aiasylum.tests.adversarial",
        adv_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {adv_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vivasecuris.aiasylum.tests.adversarial"] = mod
    spec.loader.exec_module(mod)
    return mod.AdversarialTest


def iter_hardcoded_prompts() -> Iterator[dict[str, Any]]:
    AdversarialTest = _load_adversarial_class()

    for technique, prompts in AdversarialTest.JAILBREAK_TECHNIQUES.items():
        for prompt in prompts:
            yield {
                "technique": technique,
                "prompt": prompt,
                "source": "hardcoded",
                "multi_shot": False,
            }
    for technique, pattern in AdversarialTest.MULTI_SHOT_PATTERNS.items():
        for turn in pattern.get("turns", []):
            yield {
                "technique": technique,
                "prompt": turn,
                "source": "hardcoded",
                "multi_shot": True,
            }


def iter_db_prompts(
    *,
    technique: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    try:
        from vivasecuris.aiasylum.database import get_session, PromptLibrary
    except Exception as e:  # pragma: no cover
        logger.warning("database unavailable for jailbreak export: %s", e)
        return

    session = get_session()
    try:
        rows = (
            session.query(PromptLibrary)
            .filter(PromptLibrary.category == "adversarial")
            .all()
        )
        count = 0
        for p in rows:
            if not p.tags or "jailbreak" not in p.tags:
                continue
            meta = p.meta_data or {}
            tech = meta.get("jailbreak_technique") or "unknown"
            if technique and tech != technique:
                continue
            yield {
                "technique": tech,
                "prompt": p.prompt_text,
                "source": meta.get("source") or meta.get("source_platform") or "database",
                "multi_shot": False,
            }
            count += 1
            if limit is not None and count >= limit:
                break
    finally:
        session.close()


def export_jailbreak_corpus(
    path: Path,
    *,
    include_database: bool = True,
    technique: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """Write JSONL corpus; return number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in iter_hardcoded_prompts():
            if technique and row["technique"] != technique:
                continue
            f.write(json_dumps(row) + "\n")
            written += 1
            if limit is not None and written >= limit:
                return written
        if include_database:
            remaining = None if limit is None else max(0, limit - written)
            for row in iter_db_prompts(technique=technique, limit=remaining):
                f.write(json_dumps(row) + "\n")
                written += 1
                if limit is not None and written >= limit:
                    break
    return written


def json_dumps(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False)

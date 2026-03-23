"""Warm the local Hugging Face datasets cache for every registered benchmark (minimal download).

Skipped: jailbreak (database-backed). Uses split slices like ``test[:1]`` where supported so
bootstrap does not pull full corpora.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)

_SKIP_PREFETCH = frozenset({"jailbreak"})


def _slice_split(split: str) -> str:
    s = (split or "test").strip()
    if "[" in s:
        return s
    return f"{s}[:1]"


def prefetch_benchmark_caches() -> Dict[str, str]:
    """Load one row per HF-backed benchmark. Returns benchmark_key -> status ('ok', 'skipped', or error text)."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed; skip benchmark prefetch")
        return {"_error": "datasets not installed"}

    from vivasecuris.aiasylum.benchmarks.datasets import BENCHMARK_DATASETS

    results: Dict[str, str] = {}

    for key, cfg in BENCHMARK_DATASETS.items():
        lk = key.lower()
        if lk in _SKIP_PREFETCH:
            results[key] = "skipped"
            continue
        if cfg.get("dataset") == "internal":
            results[key] = "skipped"
            continue

        dataset_name = cfg["dataset"]
        split = cfg.get("split", "test")
        dataset_config = cfg.get("config")
        slice_split = _slice_split(split)

        try:
            if dataset_config:
                try:
                    load_dataset(dataset_name, dataset_config, split=slice_split)
                except Exception:
                    load_dataset(dataset_name, name=dataset_config, split=slice_split)
            else:
                load_dataset(dataset_name, split=slice_split)
            results[key] = "ok"
            logger.info("Prefetch OK: %s (%s)", key, dataset_name)
        except Exception as e:
            msg = str(e)[:200]
            results[key] = msg
            logger.warning("Prefetch failed for %s: %s", key, e)

    return results


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = prefetch_benchmark_caches()
    failed = [k for k, v in out.items() if v not in ("ok", "skipped") and not k.startswith("_")]
    for k, v in sorted(out.items()):
        print(f"  {k}: {v}")
    if failed:
        print(
            f"\n⚠️  {len(failed)} benchmark(s) failed to prefetch "
            "(first run may still download on use)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

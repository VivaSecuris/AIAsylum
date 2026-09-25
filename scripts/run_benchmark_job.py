#!/usr/bin/env python3
"""Execute one saved benchmark row in a separately configured Python runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def main(test_run_id: int) -> None:
    from vivasecuris.aiasylum.database import TestRun, get_session
    from vivasecuris.aiasylum.runner import TestRunner
    import torch
    import transformers

    with get_session() as session:
        run = session.get(TestRun, test_run_id)
        if run is None or run.test_type != "benchmark":
            raise ValueError("Expected an existing benchmark test run")
        metadata = dict(run.meta_data or {})
        metadata["benchmark_runtime"] = {**metadata.get("benchmark_runtime", {}),
            "python_version": sys.version.split()[0], "transformers_version": transformers.__version__,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }
        run.meta_data = metadata
        session.commit()
    result = await TestRunner().execute_test_run(test_run_id)
    if result.status != "completed":
        raise RuntimeError(f"Benchmark finished with status {result.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_run_id", type=int)
    args = parser.parse_args()
    if args.test_run_id < 1:
        parser.error("test_run_id must be positive")
    os.environ["AIASYLUM_BENCHMARK_RUNTIME"] = "1"
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(args.test_run_id))

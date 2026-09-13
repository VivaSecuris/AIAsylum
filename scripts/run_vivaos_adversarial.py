#!/usr/bin/env python3
"""Run a short adversarial pass against a VivaOS Patient and emit catch JSONL.

Example:
  python scripts/run_vivaos_adversarial.py \\
    --patient-provider servus --patient-model servus \\
    --doctor-provider ollama --doctor-model llama3.2 \\
    --output catches.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vivasecuris.aiasylum.models import get_provider  # noqa: E402
from vivasecuris.aiasylum.oracle import evaluate_catch  # noqa: E402
from vivasecuris.aiasylum.tests.adversarial import AdversarialTest  # noqa: E402


async def _run(args: argparse.Namespace) -> None:
    patient_provider = get_provider(args.patient_provider)
    patient = patient_provider.create_model(args.patient_model)

    techniques = list(AdversarialTest.JAILBREAK_TECHNIQUES.keys())
    if args.technique:
        techniques = [args.technique]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for tech in techniques:
            if args.limit is not None and written >= args.limit:
                break
            test = AdversarialTest(
                name=f"vivaos-{tech}",
                technique=tech,
                max_prompts=1,
            )
            # Drive patient directly for catch scoring (doctor optional)
            for prompt in test.prompts[:1]:
                if args.limit is not None and written >= args.limit:
                    break
                resp = await patient.generate(prompt)
                catch = evaluate_catch(resp, technique=tech)
                row = {
                    **catch.to_dict(),
                    "prompt_preview": prompt[:160],
                    "response_preview": (resp.content or "")[:160],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                print(
                    f"{tech}: caught={catch.caught} reason={catch.reason} "
                    f"trace={catch.trace_id}"
                )
    print(f"Wrote {written} catch rows to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Asylum × VivaOS adversarial catch run")
    p.add_argument("--patient-provider", default="servus")
    p.add_argument("--patient-model", default="servus")
    p.add_argument("--doctor-provider", default="ollama")
    p.add_argument("--doctor-model", default="llama3.2")
    p.add_argument("--technique", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

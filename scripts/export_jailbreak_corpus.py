#!/usr/bin/env python3
"""Export jailbreak corpus (CLI-friendly script entry)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vivasecuris.aiasylum.export_corpus import export_jailbreak_corpus  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Export AIAsylum jailbreaks as JSONL")
    p.add_argument("-o", "--output", required=True, help="Output JSONL path")
    p.add_argument("--technique", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--no-database",
        action="store_true",
        help="Hardcoded techniques only",
    )
    args = p.parse_args()
    n = export_jailbreak_corpus(
        Path(args.output),
        include_database=not args.no_database,
        technique=args.technique,
        limit=args.limit,
    )
    print(f"Wrote {n} prompts to {args.output}")


if __name__ == "__main__":
    main()

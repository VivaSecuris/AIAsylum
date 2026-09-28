#!/usr/bin/env python3
"""Preview or install the versioned role/context/goal prompt catalog."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vivasecuris.aiasylum.database import get_session
from vivasecuris.aiasylum.prompt_presets import PROMPT_PRESETS, install_prompt_presets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Insert missing presets; never update existing prompts")
    mode.add_argument("--dry-run", action="store_true", help="Preview missing presets without changes (default)")
    mode.add_argument("--catalog", action="store_true", help="Print full preset texts without opening a database session")
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps([preset.as_row() for preset in PROMPT_PRESETS], indent=2))
        return 0
    with get_session() as session:
        try:
            report = install_prompt_presets(session, apply=args.apply)
            if args.apply:
                session.commit()
        except Exception:
            session.rollback()
            raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

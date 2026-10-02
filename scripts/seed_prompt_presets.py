#!/usr/bin/env python3
"""Preview or install the versioned role probes and common system prompts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vivasecuris.aiasylum.database import get_session
from vivasecuris.aiasylum.prompt_presets import CATALOG_IDS, select_prompt_presets, install_prompt_presets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Insert missing presets; never update existing prompts")
    mode.add_argument("--dry-run", action="store_true", help="Preview missing presets without changes (default)")
    mode.add_argument("--catalog", action="store_true", help="Print full preset texts without opening a database session")
    parser.add_argument("--catalog-id", choices=CATALOG_IDS, help="Limit review or installation to one catalog (default: both)")
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps([preset.as_row() for preset in select_prompt_presets(args.catalog_id)], indent=2))
        return 0
    with get_session() as session:
        try:
            report = install_prompt_presets(session, apply=args.apply, catalog_id=args.catalog_id)
            if args.apply:
                session.commit()
        except Exception:
            session.rollback()
            raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

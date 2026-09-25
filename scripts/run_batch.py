#!/usr/bin/env python3
"""Run a batch of weight-surgery stages against the API, in order, on the box.

Runs *on the remote GPU box*, under nohup, against the API on 127.0.0.1. That
placement is deliberate:

* a dropped SSH connection does not lose a run that is being paid for by the
  hour -- the batch keeps going and ``status`` picks it up;
* the API key is read from the box's own ``.env`` and never crosses a network.

A batch is a JSON file of steps. Each step is a ``POST /api/v1/weights/runs``
body plus a ``name``. Values of the form ``"$name.path.to.value"`` are resolved
from an earlier step's completed run, so a later step can consume what an
earlier one found -- e.g. the rank that ``select`` recommended::

    {"name": "sub3b", "kind": "direction", "source_model": "...", "subspace_rank": 4}
    {"name": "sel3b", "kind": "select", "source_run_id": "$sub3b.id", ...}
    {"name": "best3b", "kind": "direction", "subspace_rank": "$sel3b.metadata.summary.best.rank"}

A step with ``"expect"`` is checked against its run's summary; a failed
expectation stops the batch, because the point of a verification batch is to
stop at the first thing that is not as claimed rather than to spend more
GPU-hours building on it.

Stdlib only, so it needs nothing beyond the Python the API already runs on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = {"completed", "failed"}


def read_env_key(env_path: Path) -> str:
    for line in env_path.read_text().splitlines():
        if line.startswith("API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit(f"no API_KEYS in {env_path}")


class Api:
    def __init__(self, base: str, key: str):
        self.base = base.rstrip("/")
        self.key = key

    def call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"X-API-Key": self.key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"null")
            except ValueError:
                return e.code, None


def dig(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)]
        else:
            return None
    return obj


def resolve(value, done: dict):
    if isinstance(value, str) and value.startswith("$"):
        name, _, path = value[1:].partition(".")
        if name not in done:
            raise SystemExit(f"reference {value!r} names a step that has not run")
        out = dig(done[name], path) if path else done[name]
        if out is None:
            raise SystemExit(f"reference {value!r} resolved to nothing")
        return out
    if isinstance(value, dict):
        return {k: resolve(v, done) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, done) for v in value]
    return value


def check(expect: dict, run: dict) -> list:
    """``{"path": {"eq"|"le"|"ge"|"in": value}}`` against the run -> failures."""
    failures = []
    for path, cond in expect.items():
        got = dig(run, path)
        for op, want in cond.items():
            ok = {
                "eq": lambda: got == want,
                "le": lambda: got is not None and got <= want,
                "ge": lambda: got is not None and got >= want,
                "in": lambda: got in want,
            }[op]()
            if not ok:
                failures.append(f"{path} {op} {want!r}, got {got!r}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--out", required=True, help="Report JSON, written after every step")
    args = ap.parse_args()

    steps = json.loads(Path(args.batch).read_text())
    api = Api(args.api, read_env_key(Path(args.env)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {"batch": args.batch, "started": time.time(), "steps": []}
    done: dict = {}
    t_batch = time.time()

    def save():
        report["elapsed_s"] = round(time.time() - t_batch, 1)
        out.write_text(json.dumps(report, indent=2, default=str))

    for raw in steps:
        name = raw["name"]
        expect = raw.get("expect") or {}
        body = resolve({k: v for k, v in raw.items() if k not in ("name", "expect")}, done)
        print(f"== {name}: {body.get('kind')} ==", flush=True)
        t0 = time.time()

        status, run = api.call("POST", "/api/v1/weights/runs", body)
        if status != 200:
            print(f"   refused ({status}): {json.dumps(run)[:600]}", flush=True)
            report["steps"].append({"name": name, "refused": status, "detail": run})
            save()
            return 1

        rid = run["id"]
        last = None
        while run.get("status") not in TERMINAL:
            time.sleep(5)
            status, run = api.call("GET", f"/api/v1/weights/runs/{rid}")
            if status != 200:
                print(f"   lost the run ({status})", flush=True)
                return 1
            if run["status"] != last:
                last = run["status"]
                print(f"   run {rid}: {last}", flush=True)

        elapsed = round(time.time() - t0, 1)
        entry = {"name": name, "id": rid, "status": run["status"], "elapsed_s": elapsed,
                 "error": run.get("error"), "summary": (run.get("metadata") or {}).get("summary")}
        report["steps"].append(entry)
        done[name] = run
        print(f"   {run['status']} in {elapsed}s" + (f": {run['error']}" if run.get("error") else ""), flush=True)

        if run["status"] != "completed":
            save()
            return 1

        failures = check(expect, run)
        entry["expect_failures"] = failures
        save()
        if failures:
            print("   EXPECTATION FAILED -- stopping rather than building on it:", flush=True)
            for f in failures:
                print(f"     {f}", flush=True)
            return 2

    print(f"== batch complete in {round(time.time() - t_batch)}s ==", flush=True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Naive UTC API timestamps must agree with graph times in the user's timezone."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_api_dates_keep_utc_offsets_and_local_display_consistent():
    node = shutil.which("node")
    typescript = ROOT / "frontend/node_modules/typescript"
    if not node or not typescript.exists():
        pytest.skip("Node and frontend TypeScript dependencies are needed")
    script = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createRequire } = require('node:module');
const ts = require(process.argv[1]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
}}).outputText;
const loaded = { exports: {} };
new Function('module', 'exports', 'require', code)(loaded, loaded.exports, createRequire(process.argv[2]));
const { parseApiDate, formatDate, formatDateTime, formatRelativeDate } = loaded.exports;
assert.equal(Intl.DateTimeFormat().resolvedOptions().timeZone, 'America/Denver');
const utc = '2026-09-24T20:40:00Z';
const sameInstant = [
  '2026-09-24T20:40:00', '2026-09-24 20:40:00', '2026-09-24T20:40',
  '2026-09-24T20:40:00.000000', '2026-09-24T20:40:00+00:00',
  '2026-09-24T14:40:00-06:00', '2026-09-24T23:40:00+03:00',
];
Date.now = () => Date.parse('2026-09-24T21:40:00Z');
for (const input of sameInstant) {
  assert.equal(parseApiDate(input).getTime(), Date.parse(utc), input);
  assert.equal(formatDate(input), formatDate(utc), input);
  assert.equal(formatDateTime(input), 'Sep 24, 2026 at 2:40 PM', input);
  assert.equal(formatRelativeDate(input), formatRelativeDate(utc), input);
}
assert.equal(formatDateTime('2026-01-24T20:40:00'), 'Jan 24, 2026 at 1:40 PM');
assert.equal(parseApiDate('2026-09-24T20:40:00.123456').getTime(), Date.parse('2026-09-24T20:40:00.123Z'));
// Date objects and date-only strings retain the native JavaScript interpretation.
const date = new Date(utc);
assert.equal(parseApiDate(date), date);
assert.equal(parseApiDate('2026-09-24').getTime(), new Date('2026-09-24').getTime());
assert.equal(formatDateTime(date), formatDateTime(utc));
assert.equal(formatDate(null), '');
assert.equal(formatDateTime(undefined), '');
assert.equal(formatRelativeDate(''), '');
"""
    subprocess.run(
        [node, "-e", script, str(typescript), str(ROOT / "frontend/lib/utils.ts")],
        cwd=ROOT, env={**os.environ, "TZ": "America/Denver"},
        check=True, capture_output=True, text=True,
    )

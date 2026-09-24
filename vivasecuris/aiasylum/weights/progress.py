"""Progress reporting for the long-running weight commands.

Derivation, surgery and sweeps each run for minutes on a multi-billion
parameter model. Without output they are indistinguishable from a hang -- which
already cost one debugging cycle in this project when a stalled download looked
exactly like a stuck forward pass.

Writes to stderr so that piping stdout stays clean, and degrades to plain lines
when stderr is not a terminal (logs, CI) instead of emitting carriage returns.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Optional


def _isatty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class Reporter:
    """Step announcements with elapsed time, plus an inline counter."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.started = time.time()
        self._step_started: Optional[float] = None
        self._step_name: Optional[str] = None
        self._dirty = False

    def _write(self, text: str, newline: bool = True) -> None:
        if not self.enabled:
            return
        if self._dirty and newline:
            sys.stderr.write("\n")
            self._dirty = False
        sys.stderr.write(text + ("\n" if newline else ""))
        sys.stderr.flush()

    @contextmanager
    def step(self, name: str):
        """Announce a stage, then report how long it took."""
        self._step_name = name
        self._step_started = time.time()
        self._write(f"  {name} ...")
        try:
            yield self
        except Exception:
            self._write(f"  {name} FAILED after {format_duration(time.time() - self._step_started)}")
            raise
        else:
            self._write(f"  {name} done in {format_duration(time.time() - self._step_started)}")

    def count(self, done: int, total: int, label: str = "") -> None:
        """Inline counter; rewrites one line on a terminal, throttled otherwise."""
        if not self.enabled or not total:
            return
        pct = 100.0 * done / total
        elapsed = time.time() - (self._step_started or self.started)
        eta = ""
        if done and done < total:
            eta = f" eta {format_duration(elapsed / done * (total - done))}"
        text = f"    {label}{done}/{total} ({pct:.0f}%){eta}"
        if _isatty():
            sys.stderr.write("\r" + text.ljust(70))
            sys.stderr.flush()
            self._dirty = True
        elif done == total or done % max(1, total // 10) == 0:
            sys.stderr.write(text + "\n")
            sys.stderr.flush()

    def note(self, text: str) -> None:
        self._write(f"  {text}")

    def total_elapsed(self) -> str:
        return format_duration(time.time() - self.started)

    def as_callback(self):
        """Adapt this reporter to the ``progress=`` callable the engine expects.

        The two call sites disagree on arity, so one adapter serves both:
        ``derive_direction`` emits ``progress("capturing ...")`` for a stage and
        ``progress(None, done, total)`` for the counter inside it, while
        ``sweep_alpha`` emits ``progress(label, done, total)``. Defined here
        rather than at each call site so a subclass that redirects the output --
        to an SSE stream, say -- inherits the adapter along with it.
        """

        def report(message=None, done=None, total=None):
            if message is not None and done is None:
                self.note(message)
            elif done is not None:
                self.count(done, total or 0, f"{message} " if message else "")

        return report


def memory_report() -> dict:
    """Physical memory, free pages and swap, for a preflight check.

    Local-weights work needs the unified memory largely to itself. On a 24 GB
    Mac an Ollama model left resident takes a quarter of it, and the result is
    not an error but a silent collapse in speed: a 3B load measured 66s against
    6s for a 0.5B, and a 16-token generation exceeded seven minutes.
    """
    import re
    import subprocess

    info = {"total_gb": 0.0, "free_gb": 0.0, "swap_used_gb": 0.0, "swap_total_gb": 0.0}
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        info["total_gb"] = int(out.stdout.strip()) / 2**30
    except Exception:
        pass
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        m = re.search(r"page size of (\d+) bytes", vm)
        if m:
            page = int(m.group(1))
        free = inactive = 0
        for line in vm.splitlines():
            if line.startswith("Pages free:"):
                free = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive = int(line.split(":")[1].strip().rstrip("."))
        info["free_gb"] = (free + inactive) * page / 2**30
    except Exception:
        pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        total = re.search(r"total = ([\d.]+)M", sw)
        used = re.search(r"used = ([\d.]+)M", sw)
        if total:
            info["swap_total_gb"] = float(total.group(1)) / 1024
        if used:
            info["swap_used_gb"] = float(used.group(1)) / 1024
    except Exception:
        pass
    return info


def resident_ollama_models() -> list:
    """Models Ollama currently holds in memory, as ``(name, size)`` strings."""
    import subprocess

    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        lines = [l for l in out.stdout.splitlines()[1:] if l.strip()]
        return [(l.split()[0], " ".join(l.split()[2:4])) for l in lines]
    except Exception:
        return []


# A bf16 copy of a 3B model is roughly 6 GB; warn well before the disk fills.
LOW_DISK_GB = 15


def disk_warnings(path: str = ".", needed_gb: float = 0.0) -> list:
    """Warnings about free space before a run that writes a model.

    Lives here rather than in the CLI so the API enforces the same threshold
    instead of re-deriving one. Running out of space part-way through
    ``save_pretrained`` leaves a corrupt directory, so this is the one
    preflight that callers should treat as blocking rather than advisory.
    """
    import shutil

    warnings = []
    try:
        free_gb = shutil.disk_usage(path).free / 2**30
    except Exception:
        return warnings

    if needed_gb and free_gb < needed_gb * 1.1:
        warnings.append(
            f"Only {free_gb:.1f} GB free but roughly {needed_gb:.1f} GB is needed. "
            f"Writing a model into too little space leaves a corrupt directory."
        )
    elif free_gb < LOW_DISK_GB:
        warnings.append(
            f"Only {free_gb:.1f} GB free. A 3B model copy needs about 6 GB."
        )
    return warnings


def memory_warnings(needed_gb: float = 0.0) -> list:
    """Human-readable warnings about memory contention. Empty when all is well."""
    warnings = []
    mem = memory_report()

    for name, size in resident_ollama_models():
        warnings.append(
            f"Ollama is holding {name} ({size}) in the same unified memory. "
            f"Free it with `ollama stop {name}` before running local-weights work."
        )

    if mem["swap_total_gb"] and mem["swap_used_gb"] / mem["swap_total_gb"] > 0.8:
        warnings.append(
            f"Swap is {mem['swap_used_gb']:.0f} GB of {mem['swap_total_gb']:.0f} GB used. "
            f"The machine is thrashing; expect order-of-magnitude slowdowns."
        )

    if needed_gb and mem["free_gb"] and mem["free_gb"] < needed_gb:
        warnings.append(
            f"About {mem['free_gb']:.1f} GB free but roughly {needed_gb:.1f} GB needed. "
            f"Close other processes or use a smaller model."
        )
    return warnings

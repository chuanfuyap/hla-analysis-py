"""
Progress reporting utilities for hapy.stats.

This module provides a small progress helper that:
- prints progress with `flush=True` (works well on CI/log aggregators)
- optionally uses tqdm if installed (no hard dependency)

Usage
-----
prog = ProgressPrinter(total=N, desc="standard[AA]", use_tqdm=True)
...
prog.update(1)
...
prog.close()
"""

from __future__ import annotations
import time
from dataclasses import dataclass


def format_seconds(seconds: float) -> str:
    """
    Convert seconds into a human-readable duration string.

    Examples
    --------
    12.34  -> "12.34s"
    120.0  -> "2.00m"
    7200.0 -> "2.00h"
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds/60:.2f}m"
    return f"{seconds/3600:.2f}h"


@dataclass
class ProgressPrinter:
    """
    Progress reporter for long-running analyses.

    Parameters
    ----------
    total:
        Total number of tasks expected.
    desc:
        Short description prefix used in prints / tqdm bar.
    use_tqdm:
        If True, tries to use tqdm if installed. Falls back to prints if not available.
    print_every:
        If tqdm is not used, prints an update every N tasks (and always prints at the end).

    Notes
    -----
    This is intentionally simple and does not attempt to be a full logging framework.
    """
    total: int
    desc: str
    use_tqdm: bool = False
    print_every: int = 100

    def __post_init__(self):
        """Initialize timers and (optionally) tqdm progress bar."""
        self.start = time.perf_counter()
        self.n = 0
        self._bar = None

        if self.use_tqdm:
            try:
                from tqdm import tqdm  # optional dependency
                self._bar = tqdm(total=self.total, desc=self.desc)
            except Exception:
                self._bar = None

        print(f"{self.desc}: 0/{self.total} (starting)", flush=True)

    def update(self, inc: int = 1) -> None:
        """
        Increment progress by `inc`.

        When tqdm is enabled and available, updates tqdm.
        Otherwise prints periodic updates.
        """
        self.n += inc

        if self._bar is not None:
            self._bar.update(inc)
            return

        if self.n == self.total or (self.print_every > 0 and self.n % self.print_every == 0):
            elapsed = time.perf_counter() - self.start
            print(f"{self.desc}: {self.n}/{self.total} done (elapsed {format_seconds(elapsed)})", flush=True)

    def close(self) -> None:
        """
        Close the progress reporter and print final elapsed time.

        Safe to call even if tqdm is not available.
        """
        if self._bar is not None:
            self._bar.close()

        elapsed = time.perf_counter() - self.start
        print(f"{self.desc}: finished {self.total}/{self.total} (elapsed {format_seconds(elapsed)})", flush=True)
"""
Parallel execution helpers.

This version supports:
- thread or process executors
- true batching, meaning multiple items per submitted future
- worker initialisation, so large shared objects are sent once per worker
- streaming results as batches complete
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable, Iterator, TypeVar
from threading import local
import pickle

T = TypeVar("T")
R = TypeVar("R")

# Each executor thread/process owns its worker function.
_WORKER = local()


def _executor(backend: str):
    if backend == "process":
        return ProcessPoolExecutor
    if backend == "thread":
        return ThreadPoolExecutor
    raise ValueError(f"Unknown backend={backend!r}. Expected 'process' or 'thread'.")


def _init_worker(fn: Callable[[T], R]) -> None:
    """
    Initializer called once per worker process/thread.
    """
    _WORKER.fn = fn


def _apply_batch(batch: list[T]) -> list[R]:
    """
    Top-level batch worker, safe for process pools.
    Uses the function installed in this worker by _init_worker().
    """
    fn = getattr(_WORKER, "fn", None)
    if fn is None:
        raise RuntimeError("Worker function was not initialised.")
    return [fn(x) for x in batch]


def _chunked(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parallel_imap_batched(
    fn: Callable[[T], R],
    items: list[T],
    n_jobs: int,
    backend: str,
    *,
    batch_size: int = 32,
    max_in_flight: int | None = None,
) -> Iterator[R]:
    """
    Batched streaming map.

    Parameters
    ----------
    fn
        Per-item worker function. Must be module-scope picklable for process backend.
    items
        Items to process.
    n_jobs
        Number of workers. If <= 1, runs sequentially.
    backend
        "process" or "thread".
    batch_size
        Number of items handled inside one submitted future.
    max_in_flight
        Maximum number of batch-futures in flight at once.
        Default: 2 * n_jobs.

    Yields
    ------
    Per-item results in batch-completion order.
    """
    if n_jobs <= 1:
        for x in items:
            yield fn(x)
        return

    if max_in_flight is None:
        max_in_flight = max(1, 2 * n_jobs)
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")

    Executor = _executor(backend)
    if backend == "process":
        try:
            pickle.dumps(fn)
        except Exception as exc:
            raise ValueError(
                "Process workers require importable module-level functions and picklable "
                "analysis inputs/callbacks. Use backend='thread' for notebook-local "
                "functions or lambdas. On scripts, guard the analysis call with "
                "if __name__ == '__main__'."
            ) from exc
    batch_iter = iter(_chunked(items, batch_size))
    in_flight = set()

    with Executor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=(fn,),
    ) as ex:
        while len(in_flight) < max_in_flight:
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            in_flight.add(ex.submit(_apply_batch, batch))

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            for fut in done:
                batch_results = fut.result()
                for r in batch_results:
                    yield r

            while len(in_flight) < max_in_flight:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    break
                in_flight.add(ex.submit(_apply_batch, batch))
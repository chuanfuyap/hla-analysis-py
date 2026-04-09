from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelType = Literal["linear", "logit"]
Backend = Literal["process", "thread"]
InteractionMode = Literal["pairwise", "one_vs_block_omnibus"]


@dataclass(frozen=True)
class StandardConfig:
    """
    Configuration for standard association analysis.

    Parameters
    ----------
    model_type:
        "linear" or "logit".
    n_jobs:
        Number of parallel workers. Use 1 for sequential.
    backend:
        "process" or "thread".
    batch_size:
        Number of variants handled inside one submitted future (batch-per-future).
        Larger reduces overhead; too large reduces responsiveness and may increase tail latency.
    chunksize:
        Maximum number of *batch futures* in flight at once (submission window).
        Rule of thumb: 2*n_jobs .. 10*n_jobs.
    """
    model_type: ModelType
    n_jobs: int = 1
    backend: Backend = "process"
    batch_size: int = 32
    chunksize: int = 8  # interpreted as "max in-flight batches"


@dataclass(frozen=True)
class InteractionConfig(StandardConfig):
    """
    Configuration for interaction analysis.

    mode:
        - "pairwise": test A_col x B_col for all pairs
        - "one_vs_block_omnibus": anchor vs AA-block omnibus (AA must be one side)
    """
    mode: InteractionMode = "pairwise"


@dataclass(frozen=True)
class SurvivalConfig:
    """
    Configuration for survival analysis (Cox PH).

    n_jobs/backend/batch_size/chunksize follow same meaning as StandardConfig.
    """
    n_jobs: int = 1
    backend: Backend = "process"
    batch_size: int = 32
    chunksize: int = 8
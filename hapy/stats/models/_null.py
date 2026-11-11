"""Small, immutable null summaries; no fitted model is sent to workers."""
from dataclasses import dataclass
import hashlib

import pandas as pd


def observation_key(frame, *, include_index=True):
    """Fingerprint ordered observations, values, column order and category design.

    SHA-256 keeps cache validation compact. Category levels/order and dtypes
    matter even when observed labels happen to be identical.
    """
    design = []
    for name, dtype in zip(frame.columns, frame.dtypes):
        categories = None
        if isinstance(dtype, pd.CategoricalDtype):
            categories = (tuple(dtype.categories), dtype.ordered)
        design.append((name, str(dtype), categories))
    digest = hashlib.sha256(repr((design, str(frame.index.dtype) if include_index else None)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=include_index).to_numpy().tobytes())
    return digest.digest()


@dataclass(frozen=True)
class NullSummary:
    log_likelihood: float
    df_model: float
    model_type: str
    specification: tuple
    info: str
    sample_ids: tuple
    observations: bytes

    def matches(self, frame, model_type, specification, *, id_col=None):
        ids = tuple(frame[id_col]) if id_col is not None else tuple(frame.index)
        return (self.model_type == model_type and self.specification == specification
                and self.sample_ids == ids
                and self.observations == observation_key(frame, include_index=id_col is None))


def null_summary(frame, log_likelihood, df_model, model_type, specification, info, *, id_col=None):
    ids = tuple(frame[id_col]) if id_col is not None else tuple(frame.index)
    return NullSummary(float(log_likelihood), float(df_model), model_type, specification,
                       info, ids, observation_key(frame, include_index=id_col is None))

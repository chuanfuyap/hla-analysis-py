"""Obtain an adapter's available sample IDs without building a variant."""
import pandas as pd
from ..preprocess import sample_index, subset_samples_beagle_orientation


def genotype_sample_ids(data, requested, call_type):
    requested = sample_index(requested, source="Analysis samples")
    # The existing alignment function works on a zero-row shell: no genotype
    # values are copied, and the same column-selection conventions are used.
    columns = subset_samples_beagle_orientation(data.iloc[:0], requested, call_type).columns
    if call_type == "softcall":
        ids = requested[requested.isin(columns)]
    else:
        # Match against requested IDs rather than stripping '.1' from labels.
        # Incomplete pairs are conservatively omitted from the common baseline;
        # any different final adapter observations trigger a local null fit.
        ids = requested[requested.isin(columns) & (requested + ".1").isin(columns)]
    return pd.Index(ids).sort_values()


def null_sample_ids(adapter, hladat, requested, variant_ids):
    """Conservative omnibus eligibility and cohort lookup for built-in adapters."""
    accessor = getattr(adapter, "sample_ids", None)
    if accessor is None or not variant_ids:
        return None
    info = getattr(hladat, adapter.KIND).info
    ids = info["AA_ID"] if "AA_ID" in info else info.index
    counts = pd.Series(list(ids)).value_counts()
    if not (counts.reindex(variant_ids).fillna(0) > 1).any():
        return None
    return accessor(hladat, requested)

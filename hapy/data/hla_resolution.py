"""Resolve overlapping HLA dosage markers into non-overlapping allele dosages.

Functions
---------
resolve_hla_resolution : Retain the most specific available dosage per sample.
collapse_hla_resolution : Backward-compatible alias for the resolver.

Input has samples on rows and HLA allele markers on columns. Output contains
one sample-by-allele DataFrame per gene, with normalised allele names.
"""

import re

import numpy as np
import pandas as pd

from .hla_reference import normalise_hla_allele
from ..stats.preprocess import sample_index

__all__ = ["resolve_hla_resolution", "collapse_hla_resolution"]

_HLA_COLUMN = re.compile(r"^HLA_([A-Za-z0-9]+)_((?:\d{2}){1,5})([NLSCAQ]?)$")


def resolve_hla_resolution(df: pd.DataFrame, *, tolerance: float = 0.02,
                           invalid_totals: str = "keep") -> dict[str, pd.DataFrame]:
    """Resolve overlapping parent and descendant HLA allele dosages.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples on rows and HLA allele dosages on columns. Accepts hapy labels
        such as HLA_A_01/HLA_A_0101 and names such as A*01:01. Values must be
        nonnegative and finite, or missing. Values above two are accepted;
        fractional contributions are not clipped or rounded. Non-HLA predictors
        are ignored.
    tolerance : float, default 0.02
        Allowed deviation from 2 when evaluating resolved gene totals.
        Parent/descendant disagreement is not checked.
    invalid_totals : {'keep', 'raise', 'missing'}, default 'keep'
        Keep available resolved dosages even when totals differ from 2. This
        supports QC-filtered panels without rescaling or restoring removed
        alleles. Use 'raise' to require diploid totals, or 'missing' to set
        affected sample-gene panels to NaN. The AA converters use 'keep'.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One sample-by-allele float64 dosage matrix per gene. Allele names are
        normalised, for example A*01:01. Columns with no retained dosage are
        omitted unless needed to retain an entirely missing panel. Attributes
        record the gene, resolution policy, tolerance and invalid-total count.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        For invalid labels, duplicate identifiers, invalid dosages, or invalid
        totals when invalid_totals='raise'.

    Notes
    -----
    Descendant dosages are assigned first. Only remaining dosage is retained at
    a parent, preventing double counting. Higher-resolution dosages take
    precedence even when their sum exceeds the parent; that parent contributes
    zero additional dosage. No parent/descendant consistency check is applied.
    Every positive parent remainder is
    retained: dropping small remainders separately can lose substantial total
    dosage. Missing markers supply no additional dosage; known ancestors may
    supply the remainder. After QC removes alleles, the available dosages need
    not sum to 2; the default resolver preserves these partial panels.
    One-field remainders are retained, not assigned an arbitrary specific allele.
    Expression suffixes remain distinct terminal labels. Values are neither
    rounded to hard calls nor renormalised; chromosome phase is not inferred.

    Examples
    --------
    >>> df = pd.DataFrame({"HLA_A_0101": [2.0],
    ...                    "HLA_A_01010101": [0.7]}, index=["sample1"])
    >>> resolved = resolve_hla_resolution(df)
    >>> round(float(resolved["A"].loc["sample1", "A*01:01"]), 1)
    1.3
    """
    if invalid_totals not in {"keep", "raise", "missing"}:
        raise ValueError("invalid_totals must be 'keep', 'raise' or 'missing'")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not df.index.is_unique or not df.columns.is_unique:
        raise ValueError("Sample IDs and HLA columns must be unique")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative finite number")

    # (digits, expression_suffix, input column), grouped by HLA gene.
    groups = {}
    for column in df.columns:
        match = _HLA_COLUMN.fullmatch(str(column))
        if match is None:
            text = str(column).strip().upper()
            if "*" not in text:
                if text.startswith("HLA_"):
                    raise ValueError(f"Invalid HLA allele label {column!r}")
                continue
            name = normalise_hla_allele(text, allow_single_field=True)
            parsed = re.fullmatch(r"([A-Z0-9]+)\*(\d+(?::\d+)*)([NLSCAQ]?)", name)
            gene, fields, suffix = parsed.groups()
            if any(len(field) != 2 for field in fields.split(":")):
                raise ValueError(f"Unsupported field width in {column!r}")
            digits = fields.replace(":", "")
        else:
            gene, digits, suffix = match.groups()
        groups.setdefault(gene.upper(), []).append((digits, suffix, column))

    if not groups:
        raise ValueError("No HLA allele dosage columns found")

    df = df.copy()
    df.index = sample_index(df.index, source="HLA allele dosages")
    n = len(df)
    if not n:
        raise ValueError("The input has no samples")
    outputs = {}

    for gene, records in groups.items():
        columns = [col for _, _, col in records]
        X = df.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float64, copy=True
        )
        if np.isinf(X).any() or (X < 0).any():
            raise ValueError(f"{gene}: allele dosages must be finite and nonnegative, or missing")

        # Node key: (concatenated allele digits, optional expression suffix).
        # Unsuffixed intermediate prefixes are constructed even if not supplied.
        column_index = {}
        nodes = set()
        for i, (digits, suffix, _) in enumerate(records):
            node = (digits, suffix)
            if node in column_index:
                raise ValueError(f"{gene}: duplicate allele {digits}{suffix}")
            column_index[node] = i
            for depth in range(2, len(digits) + 1, 2):
                nodes.add((digits[:depth], ""))
            if suffix:
                nodes.add(node)

        # A suffix is a terminal allele. It is NOT a descendant of an
        # unsuffixed allele having exactly the same digits.
        def parent(node):
            digits, suffix = node
            return (digits[:-2], "") if len(digits) > 2 else None

        # Process longer names first; children always precede their parents.
        order = sorted(nodes, key=lambda node: (-len(node[0]), node[0], node[1]))
        incoming = {}  # Accumulated subtree dosage from already-processed children.
        retained = {}  # Non-overlapping mass assigned exactly at this node.
        totals = np.zeros(n, dtype=np.float64)

        for node in order:
            child_mass = incoming.pop(node, None)
            if child_mass is None:
                child_mass = np.zeros(n, dtype=np.float64)

            if node in column_index:
                observed = X[:, column_index[node]]
                known = np.isfinite(observed)
                # Higher-resolution dosage takes precedence. A smaller parent
                # contributes no additional mass and never rejects descendants.
                remainder = np.where(known, np.maximum(observed - child_mass, 0.0), 0.0)
                if np.any(remainder > 0):
                    retained[node] = remainder
            else:
                remainder = None

            mass = child_mass if remainder is None else child_mass + remainder
            ancestor = parent(node)
            if ancestor is None:
                totals += mass
            else:
                if ancestor not in incoming:
                    incoming[ancestor] = mass.copy()
                else:
                    incoming[ancestor] += mass

        bad = ~np.isfinite(totals) | (np.abs(totals - 2.0) > tolerance)
        if bad.any() and invalid_totals == "raise":
            examples = ", ".join(f"{df.index[i]}={totals[i]:.6g}"
                                 for i in np.flatnonzero(bad)[:5])
            message = (
                f"{gene}: resolved allele dosages do not sum to ~2 in "
                f"{int(bad.sum())} samples (tolerance={tolerance}). "
                f"Failing totals range from {totals[bad].min():.6g} to "
                f"{totals[bad].max():.6g}; examples: {examples}. "
            )
            raise ValueError(message + "The input may lack alleles or contain inconsistent resolutions.")


        # Most-specific names first; still preserve nonzero parent remainders.
        # Preserve observed marker states when every sample has zero/missing mass,
        # so downstream conversion can return missing rows rather than no columns.
        names = sorted(retained or column_index,
                       key=lambda node: (-len(node[0]), node[0], node[1]))
        result = pd.DataFrame(index=df.index.copy())
        for digits, suffix in names:
            allele = f"{gene}*{':'.join(digits[j:j+2] for j in range(0,len(digits),2))}{suffix}"
            values = retained.get((digits, suffix), np.zeros(n, dtype=np.float64))
            result[allele] = values
        # Entirely missing input is not an observed zero dosage.
        result.loc[~np.isfinite(X).any(axis=1), :] = np.nan
        if bad.any() and invalid_totals == "missing":
            result.loc[bad, :] = np.nan
        result.attrs.update({"gene": gene, "resolution": "highest_available_per_sample",
                             "sum_tolerance": tolerance, "N_invalid_totals": int(bad.sum())})
        outputs[gene] = result

    return outputs

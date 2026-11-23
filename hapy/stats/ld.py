"""Within-gene multiallelic amino-acid linkage disequilibrium."""
from itertools import combinations
import re

import numpy as np
import pandas as pd
from scipy.stats.contingency import association, chi2_contingency

from ..data.hla_reference import _normalise_gene


def _allele_weights(hladat, gene, reference):
    """Reuse retained allele pairs or the existing dosage-resolution pathway."""
    mapping = (hladat.attrs.get("hla_mapping") if isinstance(hladat, pd.DataFrame)
               else getattr(hladat, "hla_mapping", None))
    if mapping is not None:
        if gene not in mapping.allele_ids:
            raise ValueError(f"{gene}: gene not present in the retained allele mapping.")
        ids = mapping.allele_ids[gene]
        if not ids.size:
            raise ValueError("No HLA allele observations available.")
        weights = np.bincount(ids.ravel(), minlength=len(mapping.alleles[gene])) / ids.size
        return weights, mapping.matrices[gene], mapping.positions[gene]

    from ..data.hla_dosage import _resolved_panels, _prepare_allele_dosages
    if reference is None or not hasattr(hladat, "HLA") or hladat.type != "softcall":
        raise ValueError("LD requires a retained allele mapping or softcall HLA data and reference; "
                         "AA dosages alone do not establish within-gene phase.")
    panel = _resolved_panels(hladat.HLA.data, gene, "index", 0.02)[gene]
    X, _, samples, _, matrix, reference, gene = _prepare_allele_dosages(
        panel, reference, gene, "columns", 0.02)
    weights = X.sum(axis=0) / (2 * len(samples))
    return weights, matrix, reference.positions[gene]


def _cramers_v(joint):
    """Convert frequency weights to SciPy's required integer table."""
    # V is invariant to common scaling (correction=False). 2**52 keeps
    # absolute cell rounding to about 1e-16; it is not an inferential sample size.
    # Keep even extremely rare positive cells positive rather than truncating.
    counts = np.rint(joint * 2**52).astype(np.int64)
    counts[(joint > 0) & (counts == 0)] = 1
    value = association(counts, method="cramer", correction=False)
    # Integer rounding can distort LD when a marginal is extremely rare.
    # In that case evaluate the same uncorrected chi-square definition on
    # the original frequencies, which chi2_contingency accepts directly.
    positive = joint > 0
    restored = counts / counts.sum()
    if not np.allclose(restored[positive], joint[positive], rtol=1e-8, atol=0):
        chi2 = chi2_contingency(joint, correction=False)[0]
        value = np.sqrt(chi2 / joint.sum() / min(np.array(joint.shape) - 1))
    return float(np.clip(value, 0, 1))


def _position_key(position):
    """Order residues and their insertions: 25, 25i1, 25i2, 26."""
    match = re.fullmatch(r"(-?\d+)(?:i([1-9]\d*))?", position)
    if match is None:
        raise ValueError(f"Invalid AA position label {position!r}; expected an integer or insertion such as '25i1'.")
    return int(match[1]), int(match[2] or 0)


def LD(hladat, gene, positions=None, reference=None):
    """Calculate within-gene amino acid LD using Cramer's V (Wn).

    Parameters
    ----------
    hladat : HLAdata or pandas.DataFrame
        HLA allele dosages or AA data with a retained allele mapping.
    gene : str
        HLA gene, such as "DRB1".
    positions : iterable of int or str, optional
        AA positions to include, including insertion labels such as "25i1".
        Defaults to all available positions.
    reference : HLAReference or path, optional
        Required when no allele mapping is available.

    Returns
    -------
    pandas.DataFrame
        Symmetric Cramer's V matrix ordered by AA position.
        Monomorphic positions are excluded, and undefined values are NaN.
        Mapping and allele frequency coverage are stored in ``attrs``.

    Notes
    -----
    LD is calculated from HLA allele frequencies rather than sample-level
    dosage correlations. Missing or ambiguous residues are excluded,
    while known alignment gaps are retained. No frequency filter is applied.
    No P-values are returned.

    References
    ----------
    Thomson and Single (2014), Genetics 198:321-331.
    https://doi.org/10.1534/genetics.114.165266
    """
    gene = _normalise_gene(gene)
    weights, matrix, info = _allele_weights(hladat, gene, reference)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Allele frequencies must be finite, nonnegative and have positive total mass.")
    allele_coverage = float(weights.sum())
    weights = weights / allele_coverage
    available = {str(p): i for i, p in enumerate(info["AA_POS"])}
    if positions is None:
        requested = list(available)
    else:
        if isinstance(positions, (str, bytes)) or np.isscalar(positions):
            raise ValueError("positions must be a sequence of unique AA positions.")
        requested = [str(p) for p in positions]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("positions must be a nonempty sequence of unique AA positions.")
        missing = set(requested).difference(available)
        if missing:
            raise ValueError(f"Unknown {gene} AA positions: {sorted(missing)}")
    ordered = sorted(requested, key=_position_key)
    states, valid = {}, {}
    for pos in ordered:
        column = np.asarray(matrix[:, available[pos]], dtype=str)
        known = np.array([len(s) == 1 and s in 'ACDEFGHIKLMNPQRSTVWYOU.'
                          for s in column]) & (weights > 0)
        if len(np.unique(column[known])) > 1:
            states[pos], valid[pos] = column, known
    labels = list(states)
    values = np.eye(len(labels), dtype=float)
    coverage = np.zeros_like(values)
    for i, pos in enumerate(labels):
        coverage[i, i] = weights[valid[pos]].sum()
    for i, j in combinations(range(len(labels)), 2):
        a, b = labels[i], labels[j]
        mask = valid[a] & valid[b]
        mass = weights[mask].sum()
        coverage[i, j] = coverage[j, i] = mass
        left, li = np.unique(states[a][mask], return_inverse=True)
        right, ri = np.unique(states[b][mask], return_inverse=True)
        value = np.nan
        if mass > 0 and len(left) > 1 and len(right) > 1:
            joint = np.zeros((len(left), len(right)))
            np.add.at(joint, (li, ri), weights[mask] / mass)
            value = _cramers_v(joint)
        values[i, j] = values[j, i] = value
    index = pd.Index([p if "i" in p else int(p) for p in labels], name="AA_POS")
    result = pd.DataFrame(values, index=index, columns=index.copy())
    result.attrs.update(gene=gene, method="cramer", allele_frequency_coverage=allele_coverage,
                        mapping_coverage=pd.DataFrame(coverage, index=index, columns=index.copy()))
    return result

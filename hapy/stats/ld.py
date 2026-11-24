"""Multiallelic amino-acid linkage disequilibrium."""
from itertools import combinations
import re

import numpy as np
import pandas as pd
from scipy.linalg import svd
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


def _ld_cramer(hladat, gene, positions=None, reference=None):
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


def _dosage_basis(values):
    """Return an orthonormal basis for centred residue dosages."""
    if len(values) < 2 or values.shape[1] == 0:
        return np.empty((len(values), 0))
    centred = values - values.mean(axis=0)
    u, singular, _ = svd(centred, full_matrices=False)
    # Remove the sum-to-two dependency and any other numerical zero components.
    tolerance = np.finfo(float).eps * max(centred.shape) * max(1.0, singular[0])
    return u[:, singular > tolerance]


def _ld_cca(hladat, gene, positions):
    """Calculate pairwise dosage LD, reusing bases for shared sample sets."""
    from ..data.hla_mapping import read_aa_dosage
    from .preprocess import sample_index

    if isinstance(hladat, pd.DataFrame):
        hladat = read_aa_dosage(hladat)
    if not hasattr(hladat, "AA"):
        raise ValueError("CCA LD requires AA dosages or hard calls.")
    genes = [_normalise_gene(gene)] if isinstance(gene, str) else [_normalise_gene(g) for g in gene]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("gene must contain unique HLA genes.")
    info, raw = hladat.AA.info, hladat.AA.data
    if not raw.columns.is_unique:
        raise ValueError("AA sample columns must be unique.")
    if hladat.type == "softcall":
        samples = sample_index(raw.columns, source="AA dosages")
    else:
        # Reuse complete Beagle sample pairs without stripping dots from IIDs.
        from .adapters._samples import genotype_sample_ids
        candidates = pd.Index(raw.columns).astype(str)
        samples = genotype_sample_ids(raw, candidates, hladat.type)
    if positions is not None:
        if isinstance(positions, (str, bytes)) or np.isscalar(positions):
            raise ValueError("positions must be a sequence of unique AA positions.")
        positions = [str(pos) for pos in positions]
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("positions must be a nonempty sequence of unique AA positions.")
    matrices = {}
    for g in genes:
        selected = info["GENE"].map(_normalise_gene).eq(g) & ~info["AA_ID"].astype(str).str.startswith("MOTIF_")
        available = set(info.loc[selected, "AA_POS"].astype(str))
        if not available:
            raise ValueError(f"{g}: no AA positions available.")
        if positions is not None and set(positions).difference(available):
            raise ValueError(f"Unknown {g} AA positions: {sorted(set(positions).difference(available))}")
        for pos in sorted(available if positions is None else positions, key=_position_key):
            mask = selected & info["AA_POS"].astype(str).eq(pos)
            block = raw.iloc[np.flatnonzero(mask)]
            if hladat.type == "softcall":
                # Use residue dosages directly, without association frequency filters.
                values = block.to_numpy(dtype=float).T
            else:
                first = block.loc[:, samples].T.to_numpy()
                second = block.loc[:, samples + ".1"].T.to_numpy()
                present = ["P", "p", "T"]
                known = present + ["A", "a"]
                complete = np.isin(first, known).all(axis=1) & np.isin(second, known).all(axis=1)
                values = np.isin(first, present).astype(float) + np.isin(second, present)
                values[~complete] = np.nan
            if np.isinf(values).any() or (values < 0).any():
                raise ValueError("AA dosages must be finite and nonnegative, or missing.")
            matrices[g, pos] = values

    bases = {}
    def basis(key, mask):
        cache_key = (key, mask.tobytes())
        if cache_key not in bases:
            bases[cache_key] = _dosage_basis(matrices[key][mask])
        return bases[cache_key]

    valid = {key: np.isfinite(value).all(axis=1) for key, value in matrices.items()}
    labels = [key for key in matrices if basis(key, valid[key]).shape[1] > 0]
    values = np.eye(len(labels))
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for i, key in enumerate(labels):
        counts[i, i] = valid[key].sum()
    for i, j in combinations(range(len(labels)), 2):
        a, b = labels[i], labels[j]
        mask = valid[a] & valid[b]
        counts[i, j] = counts[j, i] = mask.sum()
        left, right = basis(a, mask), basis(b, mask)
        value = np.nan
        if left.shape[1] and right.shape[1]:
            rho = np.clip(svd(left.T @ right, compute_uv=False), 0, 1)
            value = float(np.sqrt(np.mean(rho ** 2)))
        values[i, j] = values[j, i] = value
    if len(genes) == 1:
        index = pd.Index([p if "i" in p else int(p) for _, p in labels], name="AA_POS")
    else:
        index = pd.MultiIndex.from_tuples([(g, p if "i" in p else int(p)) for g, p in labels], names=["GENE", "AA_POS"])
    result = pd.DataFrame(values, index=index, columns=index.copy())
    result.attrs.update(method="cca", gene=genes[0] if len(genes) == 1 else genes,
                        N_used=pd.DataFrame(counts, index=index, columns=index.copy()))
    return result


def LD(hladat, gene, positions=None, reference=None, method="cca"):
    """Calculate amino acid linkage disequilibrium using CCA or Cramer's V.

    Parameters
    ----------
    hladat : HLAdata or pandas.DataFrame
        AA hard calls or residue dosages, with samples as rows for DataFrames.
    gene : str or sequence of str
        HLA gene or genes to compare. Multiple genes require "cca".
    positions : iterable of int or str, optional
        AA positions to include. Defaults to all available positions.
    reference : HLAReference or path, optional
        Required for Cramer's V when no allele mapping is available.
    method : {"cca", "cramer"}, default "cca"
        CCA uses residue dosages; Cramer's V uses HLA allele frequencies.

    Returns
    -------
    pandas.DataFrame
        Symmetric LD matrix ordered by gene and AA position.
        Multiple genes use a (GENE, AA_POS) index. Monomorphic positions
        are excluded. Sample counts and coverage are stored in ``attrs``.

    Notes
    -----
    CCA measures dosage-based association without requiring phased data,
    using sqrt(mean(rho**2)) of the canonical correlations.
    Cramer's V measures haplotypic association within a gene.
    Missing data are excluded pairwise, and undefined values are NaN.
    No P-values are returned.
    """
    if method == "cca":
        return _ld_cca(hladat, gene, positions)
    if method == "cramer":
        if not isinstance(gene, str):
            raise ValueError("Cramer's V requires a single gene.")
        return _ld_cramer(hladat, gene, positions, reference)
    raise ValueError("method must be 'cca' or 'cramer'.")

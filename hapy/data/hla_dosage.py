"""Convert fractional HLA allele dosages into amino-acid or motif dosages.

Functions
---------
hla_dosage_to_aa : Calculate expected dosages for individual AA residues.
hla_dosage_to_motifs : Calculate expected within-gene AA motif copy counts.

Input matrices contain alleles on rows and samples on columns by default.
Outputs are hapy HLAdata objects with AA.data and AA.info, ready for analysis.
Use output='dataframe' for sample-by-residue or sample-by-motif dosage tables.
"""

import numpy as np
import pandas as pd

from .hla_reference import (
    HLAReference,
    _normalise_gene,
    load_hla_reference,
    normalise_hla_allele,
)
from ..stats.preprocess import sample_index
from .hla_resolution import resolve_hla_resolution


def _prepare_allele_dosages(dosages, reference, gene, allele_axis, sum_tolerance):
    """Validate diploid allele dosages and resolve reference sequences.

    Parameters
    ----------
    dosages : pandas.DataFrame
        Numeric nonnegative HLA allele dosages, or NaN. Fractional values and
        QC-filtered panels are accepted without requiring a total of two.
        Rows are alleles and columns are samples by default.
    reference : HLAReference or str or pathlib.Path
        Local reference object or path to a saved HLA reference NPZ file.
    gene : str
        HLA locus, optionally HLA-prefixed, such as DRB1.
    allele_axis : {'index', 'columns'}
        Axis containing allele labels.
    sum_tolerance : float
        Tolerance for informational comparisons of gene totals against two.
        Totals never cause rescaling or missing output.

    Returns
    -------
    tuple
        Sample-by-allele dosage array, complete-sample mask, canonical sample
        index, normalised allele names, allele-by-position sequence array,
        loaded reference and normalised gene. Missing dosages are replaced by
        zero internally; the mask preserves sample missingness for callers.

    Raises
    ------
    TypeError
        If dosages is not a DataFrame.
    ValueError
        For invalid labels, axes, genes, negative dosages or infinite values.
    """
    if not isinstance(dosages, pd.DataFrame):
        raise TypeError("dosages must be a pandas DataFrame.")
    if allele_axis not in {"index", "columns"}:
        raise ValueError("allele_axis must be 'index' or 'columns'.")
    if not isinstance(reference, HLAReference):
        reference = load_hla_reference(reference)
    gene = _normalise_gene(gene)
    if gene not in reference.alleles:
        raise ValueError(f"{gene}: gene not present in reference.")
    if not np.isfinite(sum_tolerance) or sum_tolerance < 0:
        raise ValueError("sum_tolerance must be a nonnegative finite number.")

    # Internal orientation: samples as rows and alleles as columns.
    data = dosages.T if allele_axis == "index" else dosages
    if data.empty or not data.index.is_unique or not data.columns.is_unique:
        raise ValueError("Dosage matrix must be nonempty with unique sample and allele labels.")
    sample_ids = sample_index(data.index, source="HLA allele dosages")

    allow_single = True
    names = [normalise_hla_allele(a, gene, allow_single_field=allow_single)
             for a in data.columns]
    if any(a is None for a in names) or len(set(names)) != len(names):
        raise ValueError("Allele labels must be nonmissing and unique after normalisation.")

    # Do not interpret missing allele dosages as zero.
    X = data.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(X).any() or (X < 0).any():
        raise ValueError("HLA allele dosages must be finite and nonnegative, or NaN.")
    complete = np.isfinite(X).all(axis=1)
    X = np.nan_to_num(X, nan=0.0)
    # Missing alleles are represented by '?' at every amino-acid position.
    unknown = np.full(len(reference.positions[gene]), "?", dtype="U1")
    rows = []
    for name in names:
        row = reference.resolve(name)
        rows.append(unknown if row is None else row)
    sequences = np.stack(rows)
    return X, complete, sample_ids, names, sequences, reference, gene


def _resolved_panels(dosages, gene, allele_axis, sum_tolerance):
    """Orient the input and resolve overlapping dosage markers per gene."""
    if not isinstance(dosages, pd.DataFrame):
        raise TypeError("dosages must be a pandas DataFrame.")
    if allele_axis not in {"index", "columns"}:
        raise ValueError("allele_axis must be 'index' or 'columns'.")
    data = dosages.T if allele_axis == "index" else dosages
    if gene is not None:
        gene = _normalise_gene(gene)
        def belongs(label):
            text = str(label).strip().upper().removeprefix("HLA-").removeprefix("HLA_")
            return text.startswith(gene + "*") or text.startswith(gene + "_") or ("*" not in text and "_" not in text)
        data = data.loc[:, [belongs(label) for label in data.columns]].copy()
        # Let the resolution parser decode explicit Hapy labels at every
        # resolution; the general normaliser deliberately rejects compact 8-digit names.
        data.columns = [label if "*" in str(label) or "_" in str(label)
                        else normalise_hla_allele(label, gene, allow_single_field=True)
                        for label in data.columns]
    panels = resolve_hla_resolution(data, tolerance=sum_tolerance,
                                   invalid_totals="keep")
    if gene is not None and gene not in panels:
        raise ValueError(f"{gene}: no allele dosage markers found.")
    return panels


def _check_unknown_tolerance(unknown_tolerance):
    if not np.isfinite(unknown_tolerance) or unknown_tolerance < 0:
        raise ValueError("unknown_tolerance must be a nonnegative finite number.")


def _check_output(output):
    if output not in {"hladata", "dataframe"}:
        raise ValueError("output must be 'hladata' or 'dataframe'.")


def _as_hladata(frame, reference, *, motifs=False):
    """Wrap converted dosages without applying additional QC or inventing phase."""
    from .HLAdat import HLAdata
    from .hla_mapping import read_aa_dosage

    metadata = dict(frame.attrs)
    if not motifs:
        result = read_aa_dosage(frame)
        result.AA.info["FEATURE_TYPE"] = "residue"
    else:
        gene, positions = metadata["gene"], metadata["positions"]
        group = f"MOTIF_{gene}_{'-'.join(positions)}"
        labels = list(frame.columns)
        markers = [f"{group}_{i}" for i in range(len(labels))]
        annotated = frame.T.copy()
        annotated.attrs = {}
        annotated.index = markers
        annotated["AA_ID"] = group
        annotated["TYPE"] = "AA"
        annotated["GENE"] = gene
        annotated["AA_POS"] = "|".join(positions)
        annotated["POS"] = pd.NA
        annotated["alleleA"], annotated["alleleB"] = "P", "A"
        result = HLAdata(annotated, "softcall", load_types=("AA",))
        result.AA.data.drop(columns=["alleleA", "alleleB"], inplace=True)
        result.AA.info["FEATURE_TYPE"] = "motif"
        result.AA.info["MOTIF"] = labels
        result.AA.info["POSITIONS"] = [positions] * len(labels)
        result.AA.info["IS_AMBIGUOUS"] = [label.startswith("{") for label in labels]
        resolution = metadata.get("motif_resolution", {})
        result.AA.info["IS_INFERRED"] = [any(raw != label and chosen == label
                                               for raw, chosen in resolution.items())
                                           for label in labels]
        result.hla_mapping = None
    result.reference_metadata = {"reference_version": reference.version,
                                 "assembly": reference.metadata.get("assembly"),
                                 "imgt_release": reference.metadata.get("imgt_release"),
                                 "phase_available": False}
    result.conversion_metadata = metadata
    return result


def _representative_motif(label):
    """Apply the existing AA-block heuristic per position, without inventing a motif."""
    if label is None or not label.startswith("{"):
        return label
    from ..stats.adapters._aa_haplo import checkAAblock

    candidates = [tuple(motif.split("|")) for motif in label[1:-1].split(";")]
    chosen = [checkAAblock(list(states)) for states in zip(*candidates)]
    if all(isinstance(state, str) and len(state) == 1 for state in chosen):
        motif = tuple(chosen)
        if motif in candidates:
            return "|".join(motif)
    return label


def hla_dosage_to_aa(dosages, reference, gene=None, *, allele_axis="index",
                     sum_tolerance=0.02, unknown_tolerance=1e-8, output="hladata"):
    """Convert overlapping HLA allele dosages into fractional AA state dosages.

    Parameters
    ----------
    dosages : pandas.DataFrame
        Default hladat.HLA.data format: HLA allele rows and sample columns.
        May contain multiple genes, overlapping resolutions and QC-filtered
        allele panels. All available fractional contributions are preserved;
        totals need not equal two and individual dosages may exceed two.
    reference : HLAReference or str or pathlib.Path
        Local reference object or path to a saved reference NPZ file.
    gene : str, optional
        Select one locus, such as DRB1. By default process all input genes.
    allele_axis : {'index', 'columns'}, default 'index'
        Axis containing allele labels; 'columns' means sample-by-allele input.
    sum_tolerance : float, default 0.02
        Tolerance used only to report counts of gene totals differing from two.
        It does not filter, rescale or replace any fractional values.
    unknown_tolerance : float, default 1e-8
        Maximum unresolved sequence dosage before output becomes missing.

    output : {'hladata', 'dataframe'}, default 'hladata'
        Return a hapy HLAdata object, or the sample-by-state dosage DataFrame.

    Returns
    -------
    HLAdata or pandas.DataFrame
        By default, a softcall HLAdata object with residue markers on AA.data
        rows, samples on columns, and aligned annotations in AA.info.
        conversion_metadata records genes, reference version and missing-panel
        counts. With output='dataframe', samples are rows and states are columns.
        Conflicting residues retain combined labels such as LVY; existing
        downstream ambiguous-AA handling is unchanged.

    Raises
    ------
    TypeError
        If dosages is not a DataFrame.
    ValueError
        For invalid labels, dosages, tolerances or missing reference genes.

    Notes
    -----
    Overlapping resolutions are resolved before mapping each gene. One-field
    remainders use matching reference sequences. Entirely unknown positions
    remain missing when their dosage exceeds unknown_tolerance. No chromosome
    phase is inferred. Values are not clipped, rounded, rescaled or downcast to
    float32. Totals differing from two are reported in N_invalid_totals for
    compatibility; that metadata is informational and does not mask samples.
    Missing input or unresolved sequence may still produce NaN.

    Examples
    --------
    >>> aa = hla_dosage_to_aa(hladat.HLA.data, reference)
    >>> drb1 = hla_dosage_to_aa(hladat.HLA.data, reference, gene="DRB1")
    """
    _check_output(output)
    _check_unknown_tolerance(unknown_tolerance)
    if not isinstance(reference, HLAReference):
        reference = load_hla_reference(reference)
    panels = _resolved_panels(dosages, gene, allele_axis, sum_tolerance)
    frames = []
    for locus, panel in panels.items():
        frame = _gene_dosage_to_aa(panel, reference, locus, allele_axis="columns",
                                  sum_tolerance=sum_tolerance,
                                  unknown_tolerance=unknown_tolerance)
        frame.attrs = {}
        frames.append(frame)
    result = pd.concat(frames, axis=1)
    result.attrs.update({"genes": tuple(panels), "reference_version": reference.version,
                         "dosage_type": "expected_aa_dosage",
                         "N_invalid_totals": {locus: panel.attrs["N_invalid_totals"]
                                              for locus, panel in panels.items()}})
    if len(panels) == 1:
        result.attrs["gene"] = next(iter(panels))
    return result if output == "dataframe" else _as_hladata(result, reference)


def _gene_dosage_to_aa(dosages, reference, gene, *, allele_axis="index",
                     sum_tolerance=0.02, unknown_tolerance=1e-8):
    """Map one already-resolved, non-overlapping gene panel to AA states."""
    X, complete, samples, names, matrix, reference, gene = _prepare_allele_dosages(
        dosages, reference, gene, allele_axis, sum_tolerance
    )
    frames = []
    for j, base in enumerate(reference.positions[gene]["MARKER_BASE"]):
        states = sorted(set(matrix[:, j]) - {"?"})
        if not states:
            continue
        # Indicator[a, residue] = 1 if the allele carries that residue.
        indicators = (matrix[:, j, None] == np.asarray(states)[None, :]).astype(np.float64)
        values = X @ indicators
        unknown_mass = X @ (matrix[:, j] == "?").astype(np.float64)
        values[(~complete) | (unknown_mass > unknown_tolerance), :] = np.nan
        frames.append(pd.DataFrame(values, index=samples,
                                   columns=[f"{base}_{state}" for state in states]))
    result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=samples)
    result.attrs.update({"gene": gene, "reference_version": reference.version,
                         "dosage_type": "expected_aa_dosage"})
    return result


def hla_dosage_to_motifs(dosages, reference, gene, positions, *,
                         allele_axis="index", sum_tolerance=0.02,
                         unknown_tolerance=1e-8, output="hladata"):
    """Convert HLA allele dosages into expected within-gene motif counts.

    Parameters
    ----------
    dosages : pandas.DataFrame
        Default hladat.HLA.data format: HLA allele rows and sample columns.
        May include all genes and overlapping resolutions; the selected gene
        is resolved automatically before mapping.
    reference : HLAReference or str or pathlib.Path
        Local reference object or path to a saved HLA reference NPZ file.
    gene : str
        HLA locus, optionally HLA-prefixed, such as DRB1.
    positions : iterable of str or int
        Nonempty unique AA position labels, in the order to join residues.
    allele_axis : {'index', 'columns'}, default 'index'
        Axis containing allele labels. Use 'columns' for sample-by-allele input.
    sum_tolerance : float, default 0.02
        Tolerance for informational comparisons of gene totals against two.
        Totals never cause rescaling or missing output.
    unknown_tolerance : float, default 1e-8
        Maximum dosage on unresolved sequence before the output is missing.

    output : {'hladata', 'dataframe'}, default 'hladata'
        Return a hapy HLAdata object, or the sample-by-state dosage DataFrame.

    Returns
    -------
    HLAdata or pandas.DataFrame
        By default, a softcall HLAdata object with motif markers on AA.data rows
        and samples on columns. AA.info contains MOTIF, POSITIONS, IS_AMBIGUOUS
        and IS_INFERRED annotations. All motifs share one AA_ID for analysis
        with kind='AA'. conversion_metadata retains allele groups, original
        candidate labels and missing-panel counts. With output='dataframe',
        samples are rows and motif labels are columns.

    Raises
    ------
    TypeError
        If dosages is not a DataFrame.
    ValueError
        For invalid inputs, unknown positions or empty/duplicate positions.

    Notes
    -----
    The existing checkAAblock heuristic is applied independently at each
    requested position. A representative motif is used only when each position
    resolves to one residue and the combination exists among the reference
    candidates. Ties and unsupported combinations retain their alternatives,
    such as {L|A|R;V|G|R}. This is a representative heuristic, not a posterior
    probability assignment; original alternatives remain in conversion_metadata.
    Each motif combines residues carried by one HLA allele. Dosages of alleles
    carrying the same motif are summed, allowing within-gene motif analysis
    without chromosome-specific assignments. Cross-gene phase is not inferred.
    Unresolved sequence dosage above unknown_tolerance or missing input makes
    motif values missing. Available fractional contributions otherwise remain
    unchanged, even above two or when their gene total differs from two.
    No clipping, rounding, renormalisation or float32 downcast is applied.

    Examples
    --------
    >>> motifs = hla_dosage_to_motifs(
    ...     dosages, reference, gene="DRB1", positions=[11, 13, 51]
    ... )
    """
    _check_output(output)
    _check_unknown_tolerance(unknown_tolerance)
    gene = _normalise_gene(gene)
    panels = _resolved_panels(dosages, gene, allele_axis, sum_tolerance)
    dosages, allele_axis = panels[gene], "columns"
    X, complete, samples, names, matrix, reference, gene = _prepare_allele_dosages(
        dosages, reference, gene, allele_axis, sum_tolerance
    )
    positions = tuple(str(p) for p in positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("positions must be a nonempty sequence of unique AA positions.")
    pos_lookup = {str(p): j for j, p in enumerate(reference.positions[gene]["AA_POS"])}
    missing = set(positions) - pos_lookup.keys()
    if missing:
        raise ValueError(f"Unknown {gene} AA positions: {sorted(missing)}")

    indices = [pos_lookup[p] for p in positions]
    raw_motifs = [reference.resolve_motif(name, indices) for name in names]
    motifs = [_representative_motif(label) for label in raw_motifs]
    labels = sorted({m for m in motifs if m is not None})
    indicators = np.asarray([[m == label for label in labels] for m in motifs],
                            dtype=np.float64).reshape(len(names), len(labels))
    values = X @ indicators
    unknown_mass = X @ np.asarray([m is None for m in motifs], dtype=np.float64)
    values[(~complete) | (unknown_mass > unknown_tolerance), :] = np.nan
    result = pd.DataFrame(values, index=samples, columns=labels)
    result.attrs.update({
        "gene": gene,
        "positions": positions,
        "motif_resolution": {raw: chosen for raw, chosen in zip(raw_motifs, motifs)
                             if raw is not None},
        "reference_version": reference.version,
        "N_invalid_totals": panels[gene].attrs["N_invalid_totals"],
        "allele_groups": {label: [name for name, motif in zip(names, motifs)
                                  if motif == label] for label in labels},
        "N_unresolved": int(((~complete) | (unknown_mass > unknown_tolerance)).sum()),
    })
    return result if output == "dataframe" else _as_hladata(result, reference, motifs=True)

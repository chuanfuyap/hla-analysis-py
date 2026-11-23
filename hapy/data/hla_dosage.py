"""Convert fractional HLA allele dosages into amino-acid or within-gene motif dosages.

Functions
---------
hla_dosage_to_aa : Calculate expected dosages for individual AA residues.
dosage_aa_motif : Calculate expected within-gene AA motif copy counts.

Input matrices contain alleles on rows and samples on columns by default.
Outputs are hapy HLAdata objects with AA.data and AA.info, ready for analysis.
Use output='dataframe' for sample-by-residue or sample-by-motif dosage tables.
"""

import numpy as np
import pandas as pd

from .hla_reference import (
    HLAReference,
    _ALLELE,
    _normalise_gene,
    load_hla_reference,
    normalise_hla_allele,
)
from ..stats.preprocess import sample_index
from .hla_resolution import resolve_hla_resolution
from ._mapping_io import MappingReport


def _prepare_allele_dosages(dosages, reference, gene, allele_axis, sum_tolerance, *,
                            fallback_to_two_field=False, fallbacks=None):
    """Validate allele dosages and resolve their reference sequences."""
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
        if row is None and fallback_to_two_field:
            match = _ALLELE.fullmatch(name)
            fields = match[2].split(":")
            if len(fields) > 2:
                target = gene + "*" + ":".join(fields[:2]) + match[3]
                row = reference.resolve(target)
                if row is not None and fallbacks is not None:
                    fallbacks.append({"allele": name, "mapped_as": target,
                                      "reason": "two-field consensus fallback"})
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
        # Let the resolution parser decode explicit hapy labels at every
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


def _as_hladata(frame, reference=None, *, motifs=False, mapping=None):
    """Create an HLAdata object from an amino-acid or motif dosage table."""
    from .HLAdat import HLAdata
    from .hla_mapping import read_aa_dosage

    metadata = dict(frame.attrs)
    if not motifs:
        result = read_aa_dosage(frame)
        result.AA.info.drop(columns=["EXON", "FEATURE_TYPE"], errors="ignore", inplace=True)
        result.AA.info["AA_ID"] = result.AA.info["AA_ID"].astype(str).str.replace(
            r"_exon(?:NA|\d+)$", "", regex=True)
        clean_markers = result.AA.data.index.astype(str).str.replace(r"_exon(?:NA|\d+)(?=_)", "", regex=True)
        result.AA.data.index = clean_markers
        result.AA.info.index = clean_markers
    else:
        gene, positions = metadata["gene"], metadata["positions"]
        group = metadata.get("motif_group", f"MOTIF_{gene}_{'-'.join(positions)}")
        labels = list(frame.columns)
        markers = [f"{group}_{label}" for label in labels]
        annotated = frame.T.copy()
        annotated.attrs = {}
        annotated.index = markers
        annotated["AA_ID"] = group
        annotated["TYPE"] = "AA"
        annotated["GENE"] = gene
        annotated["AA_POS"] = "-".join(positions)
        annotated["POS"] = pd.NA
        annotated["alleleA"], annotated["alleleB"] = "P", "A"
        result = HLAdata(annotated, "softcall", load_types=("AA",))
        result.AA.data.drop(columns=["alleleA", "alleleB"], inplace=True)
        result.AA.info["MOTIF"] = labels
        result.hla_mapping = None
    result.AA.data.index.name = "SNP"
    result.AA.info.index.name = "SNP"
    result.AA.data.columns.name = None
    if mapping is not None:
        result.hla_mapping = mapping
        result.hla_alleles = mapping.calls.copy(deep=True)
        result.reference_metadata = dict(mapping.metadata)
        result.reference_metadata["phase_available"] = True
    else:
        result.reference_metadata = {"reference_version": reference.version,
                                     "assembly": reference.metadata.get("assembly"),
                                     "imgt_release": reference.metadata.get("imgt_release"),
                                     "phase_available": False}
    result.conversion_metadata = metadata
    return result


def _representative_motif(label):
    """Select a representative motif using the AA-block heuristic."""
    from ._motifs import representative_motif
    return representative_motif(label)


def hla_dosage_to_aa(dosages, reference, gene=None, *, allele_axis="index",
                     sum_tolerance=0.02, unknown_tolerance=1e-8, output="hladata",
                     simpleQC=True, allele_filter=0.005, verbose=True,
                     fallback_to_two_field=True):
    """Convert HLA allele dosages into fractional amino-acid dosages.

    Parameters
    ----------
    dosages : pandas.DataFrame
        HLA allele dosages, with alleles on rows and samples on columns.
    reference : HLAReference or str or pathlib.Path
        HLA reference object or path to a saved reference file.
    gene : str or None, default None
        HLA gene to map. None maps all genes in the input.
    allele_axis : {'index', 'columns'}, default 'index'
        Axis containing the HLA allele labels.
    sum_tolerance : float, default 0.02
        Tolerance for reporting gene totals that differ from two.
    unknown_tolerance : float, default 1e-8
        Tolerance for reporting unresolved dosage.
    fallback_to_two_field : bool, default True
        Use a two-field consensus when a higher-resolution allele is absent.
    output : {'hladata', 'dataframe'}, default 'hladata'
        Output format.
    simpleQC : bool, default True
        Apply the simple MAF filter to mapped output states.
    allele_filter : float, default 0.005
        Minimum allele frequency; states must also be below 1 - allele_filter.
    verbose : bool, default True
        Print mapping, QC and timing summaries.

    Returns
    -------
    HLAdata or pandas.DataFrame
        AA.data and AA.info, or a sample-by-residue dosage DataFrame.

    Raises
    ------
    TypeError
        If dosages is not a DataFrame.
    ValueError
        For invalid labels, dosages, tolerances, output format or reference gene.
    OSError
        If the reference file cannot be read.

    Notes
    -----
    Overlapping resolutions are resolved before mapping. Fractional values are
    preserved; missing input or unavailable sequence may produce NaN.
    """
    report = MappingReport("MAPPING FRACTIONAL HLA DOSAGES TO AMINO ACIDS", simpleQC=simpleQC,
                           allele_filter=allele_filter, verbose=verbose, output=output)
    _check_output(output)
    _check_unknown_tolerance(unknown_tolerance)
    if not isinstance(reference, HLAReference):
        reference = load_hla_reference(reference)
    panels = _resolved_panels(dosages, gene, allele_axis, sum_tolerance)
    frames, resolution = [], {}
    for locus, panel in panels.items():
        frame = _gene_dosage_to_aa(panel, reference, locus, allele_axis="columns",
                                  sum_tolerance=sum_tolerance,
                                  unknown_tolerance=unknown_tolerance,
                                  fallback_to_two_field=fallback_to_two_field)
        resolution[locus] = dict(frame.attrs)
        frame.attrs = {}
        frames.append(frame)
    result = pd.concat(frames, axis=1)
    result.attrs.update({"genes": tuple(panels), "reference_version": reference.version,
                         "dosage_type": "expected_aa_dosage",
                         "resolution": resolution,
                         "N_invalid_totals": {locus: panel.attrs["N_invalid_totals"]
                                              for locus, panel in panels.items()}})
    if len(panels) == 1:
        result.attrs["gene"] = next(iter(panels))
    result = report.process(result, sum(len(panel.columns) for panel in panels.values()))
    converted = result if output == "dataframe" else _as_hladata(result, reference)
    report.finish(result)
    return converted


def _gene_dosage_to_aa(dosages, reference, gene, *, allele_axis="index",
                     sum_tolerance=0.02, unknown_tolerance=1e-8,
                     fallback_to_two_field=True):
    """Map one already-resolved, non-overlapping gene panel to AA states."""
    fallbacks = []
    X, complete, samples, names, matrix, reference, gene = _prepare_allele_dosages(
        dosages, reference, gene, allele_axis, sum_tolerance,
        fallback_to_two_field=fallback_to_two_field, fallbacks=fallbacks
    )
    frames, unmapped, partial_counts, missing_counts = [], {}, {}, {}
    observed = dosages.T if allele_axis == "index" else dosages
    complete &= ~np.array([observed.attrs.get("incomplete_input", {}).get(sample, False)
                           for sample in observed.index], dtype=bool)
    all_missing = observed.isna().all(axis=1).to_numpy()
    for j, base in enumerate(reference.positions[gene]["MARKER_BASE"]):
        states = sorted(set(matrix[:, j]) - {"?"})
        # Indicator[a, residue] = 1 if the allele carries that residue.
        indicators = (matrix[:, j, None] == np.asarray(states)[None, :]).astype(np.float64)
        values = X @ indicators
        unknown_mass = X @ (matrix[:, j] == "?").astype(np.float64)
        known_mass = X @ (matrix[:, j] != "?").astype(np.float64)
        unresolved = (~complete) | (unknown_mass > unknown_tolerance)
        missing = all_missing | (unresolved & (known_mass == 0))
        values[missing, :] = np.nan
        position = str(reference.positions[gene].iloc[j]["AA_POS"])
        unmapped[position] = dict(zip(samples, np.where(complete, unknown_mass, np.nan).tolist()))
        partial_counts[position] = int((unresolved & ~missing).sum())
        missing_counts[position] = int(missing.sum())
        if not states:
            continue
        frames.append(pd.DataFrame(values, index=samples,
                                   columns=[f"{base}_{state}" for state in states]))
    result = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=samples)
    result.attrs.update({"gene": gene, "reference_version": reference.version,
                         "dosage_type": "expected_aa_dosage",
                         "unmapped_dosage": unmapped,
                         "fallbacks": fallbacks,
                         "incomplete_input": dict(zip(samples, (~complete).tolist())),
                         "N_partial": partial_counts, "N_missing": missing_counts})
    return result


def dosage_aa_motif(dosages, reference, gene, positions, *,
                         allele_axis="index", sum_tolerance=0.02,
                         unknown_tolerance=1e-8, output="hladata", simpleQC=True,
                         allele_filter=0.005, verbose=True):
    """Convert HLA allele dosages into fractional within-gene motif dosages, i.e. input from HLA alleles load from genotype files

    Parameters
    ----------
    dosages : pandas.DataFrame
        HLA allele dosages, with alleles on rows and samples on columns.
    reference : HLAReference or str or pathlib.Path
        HLA reference object or path to a saved reference file.
    gene : str
        HLA gene to map, such as DRB1.
    positions : iterable of str or int
        Amino-acid positions in the required motif order.
    allele_axis : {'index', 'columns'}, default 'index'
        Axis containing the HLA allele labels.
    sum_tolerance : float, default 0.02
        Tolerance for reporting gene totals that differ from two.
    unknown_tolerance : float, default 1e-8
        Tolerance for reporting unmapped motif dosage.
    output : {'hladata', 'dataframe'}, default 'hladata'
        Output format.
    simpleQC : bool, default True
        Apply the simple MAF filter to mapped output states.
    allele_filter : float, default 0.005
        Minimum allele frequency; states must also be below 1 - allele_filter.
    verbose : bool, default True
        Print mapping, QC and timing summaries.

    Returns
    -------
    HLAdata or pandas.DataFrame
        AA.data and AA.info for motif analysis, or a sample-by-motif DataFrame.
        Unmapped contributions are recorded in the output metadata.

    Raises
    ------
    TypeError
        If dosages is not a DataFrame.
    ValueError
        For invalid inputs, unknown positions or empty/duplicate positions.
    OSError
        If the reference file cannot be read.

    Notes
    -----
    Motifs are formed within one gene. Mapped contributions are retained and
    unmapped dosage is reported separately. Ambiguity uses checkAAblock.
    """
    report = MappingReport("MAPPING FRACTIONAL HLA DOSAGES TO MOTIFS", simpleQC=simpleQC,
                           allele_filter=allele_filter, verbose=verbose, output=output)
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
    known_mass = X @ np.asarray([m is not None for m in motifs], dtype=np.float64)
    unresolved = (~complete) | (unknown_mass > unknown_tolerance)
    missing = unresolved & (known_mass == 0)
    values[missing, :] = np.nan
    unmapped = np.where(complete, unknown_mass, np.nan)
    result = pd.DataFrame(values, index=samples, columns=labels)
    result.attrs.update({
        "gene": gene,
        "positions": positions,
        "dosage_type": "expected_motif_dosage",
        "motif_resolution": {raw: chosen for raw, chosen in zip(raw_motifs, motifs)
                             if raw is not None},
        "reference_version": reference.version,
        "N_invalid_totals": panels[gene].attrs["N_invalid_totals"],
        "allele_groups": {label: [name for name, motif in zip(names, motifs)
                                  if motif == label] for label in labels},
        "unmapped_dosage": dict(zip(samples, unmapped.tolist())),
        "N_unresolved": int(unresolved.sum()),
        "N_partial": int((unresolved & ~missing).sum()),
        "N_missing": int(missing.sum()),
    })
    result = report.process(result, len(names))
    converted = result if output == "dataframe" else _as_hladata(result, reference, motifs=True)
    report.finish(result, motifs=True)
    return converted

"""Map within-allele insertion blocks to motif dosages."""
import re

import numpy as np
import pandas as pd

from .hla_reference import HLAReference, load_hla_reference, _normalise_gene, _ALLELE
from .hla_mapping import HLAAlleleMapping
from .hla_dosage import _resolved_panels, _prepare_allele_dosages, _as_hladata
from ._mapping_io import MappingReport


def _insertion_state(rows):
    """Return the confirmed alignment motif, status and complete alternatives."""
    if rows is None or not len(rows):
        return None, "missing", ()
    rows = np.asarray(rows)
    known = (rows != "*") & (rows != "?")
    complete = known.all(axis=1)
    candidates = sorted({"".join(row) for row in rows[complete]})
    if len(candidates) > 1:
        return None, "uncertain", candidates
    if not candidates:
        return None, "missing", ()
    candidate = np.array(list(candidates[0]))
    if not ((rows == candidate) | ~known).all():
        return None, "uncertain", tuple(candidates)
    return candidates[0], "inferred" if (~known).any() else "confirmed", tuple(candidates)


def _reference_sources(reference, gene, names):
    sources = []
    for name in names:
        match = _ALLELE.fullmatch(name)
        _, fields, suffix = match.groups()
        prefix = gene + "*" + fields
        ids = [i for i, allele in enumerate(reference.alleles[gene])
               if (allele == prefix or allele.startswith(prefix + ":") or allele == prefix + suffix)
               and (not suffix or allele.endswith(suffix))]
        if not suffix and len(fields.split(":")) == 2:
            ids = [i for i in ids if not reference.alleles[gene][i].endswith("N")]
        sources.append(reference.matrices[gene][ids] if ids else None)
    return sources


def insertion_motifs(mapping, gene, reference=None, *, allele_axis="index",
                     output="hladata", simpleQC=True, allele_filter=0.005, verbose=True):
    """Calculate insertion motif dosages from HLA alleles.

    Parameters
    ----------
    mapping : HLAdata, HLAAlleleMapping or pandas.DataFrame
        Paired allele mapping, or fractional allele dosages.
    gene : str
        HLA gene containing insertion columns.
    reference : HLAReference or path or None, default None
        Required for fractional allele dosages.
    allele_axis : {'index', 'columns'}, default 'index'
        Allele axis of a fractional dosage DataFrame.
    output : {'hladata', 'dataframe'}, default 'hladata'
        HLAdata or a sample-by-insertion-marker table.
    simpleQC : bool, default True
        Apply the simple MAF filter after mapping.
    allele_filter : float, default 0.005
        Minimum frequency; states must also be below 1 - allele_filter.
    verbose : bool, default True
        Print mapping, QC and timing summaries.

    Returns
    -------
    HLAdata or pandas.DataFrame
        Insertion dosages with block resolution metadata.

    Raises
    ------
    TypeError
        If mapping is not a supported input.
    ValueError
        For invalid input, missing reference or absent insertion columns.
    OSError
        If the reference file cannot be read.

    Notes
    -----
    Absence contributes zero. Uncertain contributions remain unassigned;
    confirmed partial dosages are preserved without normalisation.
    """
    report = MappingReport("MAPPING HLA INSERTION MOTIFS", simpleQC=simpleQC,
                           allele_filter=allele_filter, verbose=verbose, output=output)
    gene = _normalise_gene(gene)
    phase = (mapping if isinstance(mapping, HLAAlleleMapping) else
             mapping.attrs.get("hla_mapping") if isinstance(mapping, pd.DataFrame) else
             getattr(mapping, "hla_mapping", None))
    if phase is not None:
        if gene not in phase.positions:
            raise ValueError(f"{gene}: gene was not mapped.")
        positions = phase.positions[gene]
        names = phase.alleles[gene]
        if gene not in phase.motif_sources:
            raise ValueError("Original candidate sequences are required; remap allele calls.")
        sources = phase.motif_sources[gene]
        samples = phase.calls.index.copy()
        ids = phase.allele_ids[gene]
        X = np.zeros((len(samples), len(names)), dtype=float)
        np.add.at(X, (np.repeat(np.arange(len(samples)), 2), ids.ravel()), 1.0)
        complete = np.ones(len(samples), dtype=bool)
        all_missing = np.zeros(len(samples), dtype=bool)
        version = phase.metadata["reference_version"]
        n_hla = sum(bool(name) for name in names)
    elif isinstance(mapping, pd.DataFrame):
        if reference is None:
            raise ValueError("reference is required for fractional allele dosages.")
        if not isinstance(reference, HLAReference):
            reference = load_hla_reference(reference)
        panel = _resolved_panels(mapping, gene, allele_axis, 0.02)[gene]
        X, complete, samples, names, _, reference, gene = _prepare_allele_dosages(
            panel, reference, gene, "columns", 0.02)
        complete &= ~np.array([panel.attrs.get("incomplete_input", {}).get(sample, False)
                               for sample in panel.index], dtype=bool)
        all_missing = panel.isna().all(axis=1).to_numpy()
        positions = reference.positions[gene]
        sources = _reference_sources(reference, gene, names)
        version = reference.version
        n_hla = len(names)
    else:
        raise TypeError("mapping must contain paired allele mapping or fractional allele dosages.")

    blocks = {}
    for j, position in enumerate(positions["AA_POS"].astype(str)):
        match = re.fullmatch(r"(-?\d+)i([1-9]\d*)", position)
        if match:
            blocks.setdefault(match[1], []).append((int(match[2]), j, position))
    if not blocks:
        raise ValueError(f"{gene}: no insertion columns in the reference.")

    frames, block_metadata = [], {}
    for anchor, columns in sorted(blocks.items(), key=lambda item: int(item[0])):
        columns.sort()
        indices = [j for _, j, _ in columns]
        group = f"MOTIF_{gene}_{anchor}ins"
        assignments, statuses, alternatives = [], [], []
        for source in sources:
            selected = None if source is None else np.asarray(source)[:, indices]
            state, status, candidates = _insertion_state(selected)
            assignments.append(None if state is None else state.replace(".", ""))
            statuses.append(status)
            alternatives.append(candidates)
        labels = sorted({candidate.replace(".", "") for candidates in alternatives
                         for candidate in candidates if candidate.replace(".", "")})
        indicators = np.array([[state == label for label in labels] for state in assignments],
                              dtype=float).reshape(len(names), len(labels))
        values = X @ indicators
        assignable = np.array([state is not None for state in assignments])
        known_mass = X @ assignable.astype(float)
        uncertain_mass = X @ np.array([s == "uncertain" for s in statuses], dtype=float)
        missing_mass = X @ np.array([s == "missing" for s in statuses], dtype=float)
        unresolved = (~complete) | ((uncertain_mass + missing_mass) > 1e-8)
        missing = all_missing | (unresolved & (known_mass == 0))
        values[missing] = np.nan
        frame = pd.DataFrame(values, index=samples, columns=[f"{group}_{label}" for label in labels])
        frames.append(frame)
        block_metadata[group] = {
            "positions": tuple(p for _, _, p in columns), "anchor": anchor,
            "allele_resolution": {str(name): {"motif": state, "status": status,
                                               "alternatives": list(candidates)}
                                  for name, state, status, candidates in
                                  zip(names, assignments, statuses, alternatives)},
            "uncertain_dosage": dict(zip(samples, uncertain_mass.tolist())),
            "unmapped_dosage": dict(zip(samples, np.where(complete, missing_mass, np.nan).tolist())),
            "absence_dosage": dict(zip(samples, (X @ np.array([s == "" for s in assignments], dtype=float)).tolist())),
            "N_partial": int((unresolved & ~missing).sum()), "N_missing": int(missing.sum()),
        }
    result = pd.concat(frames, axis=1)
    result.attrs.update({"gene": gene, "reference_version": version,
                         "dosage_type": "insertion_motif_dosage", "insertion_blocks": block_metadata,
                         "incomplete_input": dict(zip(samples, (~complete).tolist()))})
    result = report.process(result, n_hla)
    if output == "hladata":
        converted = []
        for group, details in block_metadata.items():
            selected = result.loc[:, result.columns.str.startswith(group + "_")].copy()
            selected.columns = [marker[len(group) + 1:] for marker in selected.columns]
            selected.attrs = {"gene": gene, "positions": details["positions"], "motif_group": group}
            converted.append(_as_hladata(selected, reference, motifs=True, mapping=phase))
        obj = converted[0]
        obj.AA.data = pd.concat([part.AA.data for part in converted], axis=0)
        obj.AA.info = pd.concat([part.AA.info for part in converted], axis=0)
        obj.conversion_metadata = dict(result.attrs)
        report.finish(result, motifs=True)
        return obj
    report.finish(result, motifs=True)
    return result

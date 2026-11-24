"""Shared preparation for within-gene motif models."""
from functools import partial

import pandas as pd


def _choose_motif_builder(hladat, reference, index):
    """Choose one established builder for the whole search, with per-run caches."""
    from copy import copy
    from ..data.hla_mapping import mapped_aa_motif
    from ..data.hla_dosage import dosage_aa_motif
    from ..data.hla_reference import HLAReference, load_hla_reference

    mapping = getattr(hladat, "hla_mapping", None)
    if mapping is not None:
        mapping = copy(mapping)
        mapping.motif_cache = dict(mapping.motif_cache)
        return partial(mapped_aa_motif, mapping, output="dataframe",
                       simpleQC=False, verbose=False)
    if reference is None:
        raise ValueError("Motif selection requires a stored allele mapping or HLA dosages and reference.")
    reference = reference if isinstance(reference, HLAReference) else load_hla_reference(reference)
    reference = copy(reference)
    reference._resolved = dict(reference._resolved)
    reference._motif_rows = {}
    if not hasattr(hladat, "HLA") or hladat.type != "softcall":
        raise ValueError("Load the original softcall HLA block in hladat; "
                         "AA dosages alone cannot recover phase.")
    if not isinstance(hladat.HLA.data, pd.DataFrame):
        raise TypeError("hladat.HLA.data must be a DataFrame with alleles on rows and samples on columns.")
    dosage = hladat.HLA.data.copy()
    dosage.columns = dosage.columns.astype(str)
    if not dosage.columns.is_unique:
        raise ValueError("HLA dosage sample IDs must be unique.")
    dosage = dosage.reindex(columns=index)
    return partial(dosage_aa_motif, dosage, reference, output="dataframe",
                   simpleQC=False, verbose=False)


def _build_motifs(builder, gene, positions, index):
    """Keep observed groups and their allele assignments, before reference dropping."""
    frame = builder(gene=gene, positions=positions).reindex(index)
    observed = frame.sum(axis=0, min_count=1).gt(0)
    return frame.loc[:, observed].copy()


def _expanded_motif_parents(old_motifs, new_motifs):
    """An expanded group must refine exactly one existing allele group."""
    old_groups = old_motifs.attrs["allele_groups"]
    parent_by_allele = {allele: motif for motif, alleles in old_groups.items() for allele in alleles}
    parents = {}
    for motif in new_motifs:
        alleles = new_motifs.attrs["allele_groups"][motif]
        assigned = {parent_by_allele.get(allele) for allele in alleles}
        if len(assigned) != 1 or None in assigned:
            return None
        parent = next(iter(assigned))
        if parent not in old_motifs:
            return None
        parents[motif] = parent
    return parents


def _motif_genotype(frame, prefix):
    """Drop the chosen reference and use safe formula names, preserving labels."""
    columns = [c for c in frame if c != frame.attrs["reference"]]
    names = {c: f"{prefix}_{i}" for i, c in enumerate(columns)}
    geno = frame[columns].rename(columns=names)
    return geno, {names[c]: c for c in columns}


def _initial_motifs(builder, gene, positions, index):
    """Build a starting motif, keeping the reference consistent as positions are added."""
    old_motifs = None
    # Keep the same reference choices when continuing a previous search.
    # Only prefixes of the supplied conditions are built; no selection scans run.
    for length in range(1, len(positions) + 1):
        motifs = _build_motifs(builder, gene, positions[:length], index)
        if len(motifs.columns) < 2:
            raise ValueError("Selected positions have insufficient motif variation.")
        if old_motifs is None:
            reference_motif = motifs.sum().idxmax()
        else:
            parents = _expanded_motif_parents(old_motifs, motifs)
            children = [] if parents is None else [c for c, parent in parents.items()
                if parent == old_motifs.attrs["reference"]]
            if not children:
                raise ValueError("Cannot keep the reference consistent across the supplied motif positions.")
            reference_motif = motifs[children].sum().idxmax()
        motifs.attrs["reference"] = reference_motif
        old_motifs = motifs

    return old_motifs


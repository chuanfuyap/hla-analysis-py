"""
Reusable filters for standard and interaction analyses.

Filter contexts
---------------
Standard / single-variant filters receive VariantCtx:
  {
    "analysis": "standard",
    "kind": "AA"|"HLA"|"SNP",
    "variant_id": "<variant id>",
    "meta": {...},
    "geno_cols": [...]
  }

Interaction filters receive PairCtx:
  {
    "analysis": "interaction",
    "a_kind": "...",
    "b_kind": "...",
    "a_id": "...",
    "b_id": "...",
    "a_meta": {...},
    "b_meta": {...},
    # plus either (pairwise):
    "a_col": "...",
    "b_col": "..."
    # or (omnibus):
    "anchor_col": "...",
    "block_cols": [...]
  }

All filters must return True to keep the task, False to drop it.
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, TypedDict, NotRequired


class VariantCtx(TypedDict):
    analysis: str
    kind: str
    variant_id: str
    meta: dict[str, Any]
    geno_cols: NotRequired[list[str]]


class PairCtx(TypedDict):
    analysis: str
    a_kind: str
    b_kind: str
    a_id: str
    b_id: str
    a_meta: dict[str, Any]
    b_meta: dict[str, Any]
    a_col: NotRequired[str]
    b_col: NotRequired[str]
    anchor_col: NotRequired[str]
    block_cols: NotRequired[list[str]]


VariantFilter = Callable[[VariantCtx], bool]
PairFilter = Callable[[PairCtx], bool]


def _norm_set(vals: Iterable[str] | None) -> set[str] | None:
    if vals is None:
        return None
    return {str(v) for v in vals}


def _pick_gene(meta: dict[str, Any]) -> str | None:
    for k in ("GENE", "gene", "Gene"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


def compose_variant_filters(*filters: VariantFilter) -> VariantFilter:
    """Combine multiple VariantFilters with logical AND."""
    def _f(ctx: VariantCtx) -> bool:
        return all(f(ctx) for f in filters)
    return _f


def compose_pair_filters(*filters: PairFilter) -> PairFilter:
    """Combine multiple PairFilters with logical AND."""
    def _f(ctx: PairCtx) -> bool:
        return all(f(ctx) for f in filters)
    return _f


# ---- Variant filters (standard) ----

def only_genes_variant(genes: Iterable[str]) -> VariantFilter:
    """
    Keep only variants whose meta['GENE'] is in `genes`.

    Notes
    -----
    SNPAdapter meta does not include GENE by default, so this filter will
    drop SNPs unless you extend SNPAdapter to add gene annotation.
    """
    gs = _norm_set(genes)

    def _f(ctx: VariantCtx) -> bool:
        g = _pick_gene(ctx.get("meta", {}) or {})
        return (g in gs) if gs is not None else True

    return _f


def only_variants_variant(variant_ids: Iterable[str]) -> VariantFilter:
    """Keep only variants whose variant_id is in `variant_ids`."""
    vs = _norm_set(variant_ids)

    def _f(ctx: VariantCtx) -> bool:
        return str(ctx["variant_id"]) in vs if vs is not None else True

    return _f


# ---- Pair filters (interaction) ----

def only_genes_pair(genes_a: Iterable[str] | None = None, genes_b: Iterable[str] | None = None) -> PairFilter:
    """
    Keep only interaction tasks where A and/or B meta['GENE'] matches.

    If a side does not have 'GENE' in meta, it will be treated as missing and will fail the filter.
    """
    ga = _norm_set(genes_a)
    gb = _norm_set(genes_b)

    def _f(ctx: PairCtx) -> bool:
        if ga is not None:
            g = _pick_gene(ctx.get("a_meta", {}) or {})
            if g is None or g not in ga:
                return False
        if gb is not None:
            g = _pick_gene(ctx.get("b_meta", {}) or {})
            if g is None or g not in gb:
                return False
        return True

    return _f


def only_variants_pair(
    a_allow: Iterable[str] | None = None,
    b_allow: Iterable[str] | None = None,
    *,
    use_gene_for_non_snp: bool = True,
) -> PairFilter:
    """
    Keep only interaction tasks matching allow-lists on each side.

    Matching rule (as requested)
    ----------------------------
    - If kind is "SNP": match by variant id (a_id/b_id)
    - Else (AA/HLA):
        if use_gene_for_non_snp=True: match by meta['GENE']
        else: match by variant id

    This is most useful when:
    - A is SNP: allow-list is SNP IDs
    - B is AA/HLA: allow-list is gene names
    """
    aa = _norm_set(a_allow)
    bb = _norm_set(b_allow)

    def _key(kind: str, variant_id: str, meta: dict[str, Any]) -> str | None:
        if kind == "SNP":
            return str(variant_id)
        return _pick_gene(meta) if use_gene_for_non_snp else str(variant_id)

    def _f(ctx: PairCtx) -> bool:
        if aa is not None:
            k = _key(ctx["a_kind"], ctx["a_id"], ctx.get("a_meta", {}) or {})
            if k is None or k not in aa:
                return False
        if bb is not None:
            k = _key(ctx["b_kind"], ctx["b_id"], ctx.get("b_meta", {}) or {})
            if k is None or k not in bb:
                return False
        return True

    return _f
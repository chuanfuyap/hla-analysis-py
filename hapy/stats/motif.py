"""Association tests for specified amino acid motifs and motif expansions."""
import numpy as np
import pandas as pd

from .types import StandardConfig
from .runners.standard_runner import _clean_output
from .preprocess import prepare_y, prepare_covar, make_model_table
from .adapters.aa import aa_frequency
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from ._motif import (_choose_motif_builder, _build_motifs, _initial_motifs,
                     _expanded_motif_parents, _motif_genotype)


def _motif_inputs(hladat, config, positions, add_position, famfile, covar, y):
    """Validate AA variant IDs and prepare phenotype and covariates."""
    if not isinstance(config, StandardConfig) or config.model_type not in ("linear", "logit"):
        raise ValueError("config must be StandardConfig with model_type='linear' or 'logit'.")
    if not hasattr(hladat, "AA"):
        raise ValueError("AA not loaded.")
    if isinstance(positions, (str, bytes)):
        raise ValueError("positions must be a nonempty sequence of AA variant IDs.")
    try:
        positions = list(positions)
    except TypeError as exc:
        raise ValueError("positions must be a nonempty sequence of AA variant IDs.") from exc
    if not positions or any(not isinstance(v, str) for v in positions):
        raise ValueError("positions must be a nonempty sequence of AA variant IDs.")
    variants = positions + ([add_position] if add_position is not None else [])
    if len(set(variants)) != len(variants):
        raise ValueError("Select unique AA positions; add_position must not already be included.")
    annotations = []
    for variant in variants:
        if not isinstance(variant, str) or variant.startswith("MOTIF_"):
            raise ValueError(f"Invalid AA position: {variant}")
        rows = hladat.AA.info.loc[hladat.AA.info["AA_ID"].eq(variant)]
        labels = rows[["GENE", "AA_POS"]].drop_duplicates()
        if len(labels) != 1 or labels.isna().any().any():
            raise ValueError(f"Invalid or ambiguous AA position: {variant}")
        annotations.append(tuple(labels.iloc[0].astype(str)))
    if len({gene for gene, _ in annotations}) != 1:
        raise ValueError("Motif positions must belong to one HLA gene.")
    if len(set(annotations)) != len(annotations):
        raise ValueError("Variant IDs must identify distinct AA positions.")
    phenotype = prepare_y(famfile, y=y)
    covariates = prepare_covar(covar, phenotype.index)
    # Y is reserved for the phenotype, not an adjustment variable.
    if covariates is not None and "Y" in covariates:
        raise ValueError("Covariates conflict with reserved column Y.")
    return positions, annotations[0][0], [p for _, p in annotations], phenotype, covariates


def _motif_row(gene, positions, variants, motifs, conditions=(), reduced_count=0):
    """Describe the specified motif model without stepwise selection fields."""
    return dict(GENE=gene, POSITIONS=";".join(variants),
                MOTIF_POSITIONS="-".join(positions), COND_KIND="AA",
                COND_VARIANTS=";".join(conditions), N_COND_VARIANTS=len(conditions),
                N_REDUCED_MOTIFS=reduced_count, N_EXPANDED_MOTIFS=len(motifs.columns),
                TEST="MOTIF", **{"P-value": np.nan},
                Amino_Acids=", ".join(str(c) for c in motifs.columns),
                AAcount=len(motifs.columns), AA_AF=np.nan, Ref_AA=np.nan,
                Beta=np.nan, StdErr=np.nan, Statistic=np.nan, DoF=np.nan, INFO=np.nan)


def analyse_motif(hladat, config: StandardConfig, positions, *, famfile=None,
                  covar=None, y=None, reference=None, check_rank=False) -> pd.DataFrame:
    """Test a specified amino acid motif against covariates alone.

    Null:
      Y ~ covars
    Alt:
      Y ~ covars + motif(positions)

    Parameters
    ----------
    hladat : HLAdata
        Retained allele mapping, or HLA allele dosages with a reference.
    config : StandardConfig
        Configuration with model_type="linear" or "logit".
    positions : sequence of str
        Ordered AA variant IDs from hladat.AA.info["AA_ID"], within one gene.
        One or more positions can be supplied.
    famfile : pandas.DataFrame, optional
        Sample information containing IID and PHENO when y is not supplied.
    covar : pandas.DataFrame or path, optional
        Covariates indexed by IID or containing an IID column.
    y : pandas.Series or array-like, optional
        Phenotype indexed by IID, or aligned to famfile row order.
        Without famfile, y must be a Series indexed by IID.
    reference : HLAReference or path, optional
        Required when no retained allele mapping is available.
    check_rank : bool, optional
        Check for additional model rank before fitting. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        One row with motif positions, labels and counts, sample/QC counts,
        likelihood-ratio P-value, statistic, degrees of freedom, coefficients,
        standard errors and INFO. Unusable fits are described in INFO.

    Notes
    -----
    Both models use the same complete-case samples. The reference is kept
    consistent as positions are added in the supplied order. This tests a fixed motif, without selection.
    AA dosages alone cannot recover motifs.
    """
    variants, gene, labels, phenotype, covariates = _motif_inputs(
        hladat, config, positions, None, famfile, covar, y)
    builder = _choose_motif_builder(hladat, reference, phenotype.index)
    try:
        motifs = _initial_motifs(builder, gene, labels, phenotype.index)
    except ValueError as exc:
        if str(exc) != "Selected positions have insufficient motif variation.":
            raise
        motifs = _build_motifs(builder, gene, labels, phenotype.index)
        row = _motif_row(gene, labels, variants, motifs)
        row["INFO"] = str(exc)
        return _clean_motif_output(pd.DataFrame([row]))
    row = _motif_row(gene, labels, variants, motifs)
    # All models use the same complete-case samples.
    expanded, labels = _motif_genotype(motifs, "MOTIF_F")
    if covariates is not None and not expanded.columns.intersection(covariates.columns).empty:
        raise ValueError("Covariate names collide with internal motif column names.")
    abt, qc = make_model_table(expanded, phenotype, covariates)
    row.update(qc, Ref_AA=motifs.attrs["reference"], AA_AF=aa_frequency(motifs, abt.index))
    if abt.empty:
        row["INFO"] = "no complete-case samples"
    else:
        row = _fit_motif(abt, [], list(expanded), covariates, config.model_type,
                         row, check_rank=check_rank, labels=labels)
    return _clean_motif_output(pd.DataFrame([row]))


def expand_motif(hladat, config: StandardConfig, positions, add_position, *,
                 famfile=None, covar=None, y=None, reference=None,
                 check_rank=False, _prepared=None) -> pd.DataFrame:
    """Test whether an additional position improves an existing motif.

    Null:
      Y ~ covars + motif(positions)
    Alt:
      Y ~ covars + motif(positions + add_position)

    Parameters
    ----------
    hladat : HLAdata
        Retained allele mapping, or HLA allele dosages with a reference.
    config : StandardConfig
        Configuration with model_type="linear" or "logit".
    positions : sequence of str
        Ordered AA variant IDs defining the existing motif, within one gene.
    add_position : str
        New AA variant ID from the same gene, not already in positions.
    famfile : pandas.DataFrame, optional
        Sample information containing IID and PHENO when y is not supplied.
    covar : pandas.DataFrame or path, optional
        Covariates indexed by IID or containing an IID column.
    y : pandas.Series or array-like, optional
        Phenotype indexed by IID, or aligned to famfile row order.
        Without famfile, y must be a Series indexed by IID.
    reference : HLAReference or path, optional
        Required when no retained allele mapping is available.
    check_rank : bool, optional
        Check for additional model rank before fitting. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        One row with motif positions, labels and counts, sample/QC counts,
        likelihood-ratio P-value, statistic, degrees of freedom, coefficients,
        standard errors and INFO. Unusable expansions are described in INFO.

    Notes
    -----
    Both models use the same complete-case samples. Expanded groups must
    reconstruct the existing motif dosages. The reference is kept consistent
    as positions are added in the supplied order, matching motif stepwise.
    """
    if _prepared is None:
        if not isinstance(add_position, str):
            raise ValueError("add_position must be an AA variant ID.")
        variants, gene, labels, y, covar = _motif_inputs(
            hladat, config, positions, add_position, famfile, covar, y)
        builder = _choose_motif_builder(hladat, reference, y.index)
        try:
            old_motifs = _initial_motifs(builder, gene, labels[:-1], y.index)
        except ValueError as exc:
            if str(exc) != "Selected positions have insufficient motif variation.":
                raise
            motifs = _build_motifs(builder, gene, labels, y.index)
            row = _motif_row(gene, labels, variants + [add_position], motifs, conditions=variants)
            row.update(VARIANT=add_position, AA_POS=labels[-1], INFO=str(exc))
            return _clean_motif_output(pd.DataFrame([row]))
        null_cache = None
        variant_filter = None
    else:
        # Stepwise supplies validated data and its existing builder, motifs and null fit.
        variants = list(positions)
        gene = _prepared["gene"]
        labels = _prepared["positions"] + [_prepared["position"]]
        builder = _prepared["builder"]
        old_motifs = _prepared["old_motifs"]
        null_cache = _prepared["null_cache"]
        variant_filter = _prepared["variant_filter"]

    new_motifs = _build_motifs(builder, gene, labels, y.index)
    row = _motif_row(gene, labels, variants + [add_position], new_motifs,
                     conditions=variants, reduced_count=len(old_motifs.columns))
    row.update(VARIANT=add_position, AA_POS=labels[-1])

    # Expanded groups must belong to exactly one existing motif group.
    parents = _expanded_motif_parents(old_motifs, new_motifs)
    if parents is None:
        row["INFO"] = "expanded motif groups do not refine the existing groups"
        return _motif_result(row, None, _prepared)

    children = [c for c, parent in parents.items() if parent == old_motifs.attrs["reference"]]
    if not children:
        row["INFO"] = "no observed child of the reduced reference motif"
        return _motif_result(row, None, _prepared)

    # Drop a child of the old reference so the reduced model stays nested.
    reference_motif = new_motifs[children].sum().idxmax()
    if _prepared is not None and "reference_motif" in _prepared:
        reference_motif = _prepared["reference_motif"]
        if reference_motif not in children:
            row["INFO"] = "expanded reference does not belong to the existing reference group"
            return _motif_result(row, None, _prepared)
    new_motifs.attrs["reference"] = reference_motif
    row["Ref_AA"] = new_motifs.attrs["reference"]
    reduced, _ = _motif_genotype(old_motifs, "MOTIF_R")
    expanded, labels = _motif_genotype(new_motifs, "MOTIF_F")

    if variant_filter is not None:
        ctx = dict(analysis="standard", kind="AA", variant_id=add_position,
                   meta=_prepared["meta"], geno_cols=list(expanded))
        if not variant_filter(ctx):
            row["INFO"] = "excluded by variant filter"
            return _motif_result(row, None, _prepared)

    predictors = pd.concat([reduced, expanded], axis=1)
    if covar is not None and not predictors.columns.intersection(covar.columns).empty:
        raise ValueError("Covariate names collide with internal motif column names.")
    abt, qc = make_model_table(predictors, y, covar)
    row.update(qc)
    row["AA_AF"] = aa_frequency(new_motifs, abt.index)
    if abt.empty:
        row["INFO"] = "no complete-case samples"
        return _motif_result(row, None, _prepared)

    # Check the dosages too: group labels alone do not prove nesting.
    for parent in old_motifs:
        children = [c for c, old in parents.items() if old == parent]
        recovered = new_motifs.loc[abt.index, children].sum(axis=1)
        if not np.allclose(recovered, old_motifs.loc[abt.index, parent], rtol=1e-8, atol=1e-8):
            row["INFO"] = "expanded dosages do not reconstruct existing motif groups"
            return _motif_result(row, None, _prepared)

    row = _fit_motif(abt, list(reduced), list(expanded), covar, config.model_type,
                     row, null_cache=null_cache, check_rank=check_rank, labels=labels)
    return _motif_result(row, new_motifs, _prepared)


def _motif_result(row, motifs, prepared):
    """Keep fitted motif matrices available to the stepwise search."""
    if prepared is not None:
        # Stepwise adds its own metadata and selection fields.
        row.pop("POSITIONS")
        for column in ("Beta", "StdErr", "Statistic", "DoF"):
            if pd.isna(row[column]):
                row.pop(column)
        return row, motifs
    return _clean_motif_output(pd.DataFrame([row]))


def _fit_motif(abt, reduced, expanded, covar, model_type, row, *,
               null_cache=None, check_rank=False, labels=None):
    """Fit the nested models and describe numerical failures in INFO."""
    from .models.standard import fit_motif_omnibus

    try:
        row.update(fit_motif_omnibus(abt, reduced, expanded,
            list(covar) if covar is not None else [], model_type,
            null_cache=null_cache, check_rank=check_rank, term_labels=labels))
    except (PerfectSeparationError, np.linalg.LinAlgError, FloatingPointError,
            OverflowError, ZeroDivisionError) as exc:
        row["INFO"] = f"numerical fit failure: {type(exc).__name__}: {exc}"
    except ValueError as exc:
        if not any(message in str(exc).lower() for message in (
                "first guess on the deviance function returned a nan",
                "nan, inf or invalid value detected in weights")):
            raise
        row["INFO"] = f"numerical fit failure: ValueError: {exc}"
    return row


def _clean_motif_output(dataframe):
    """Clean motif results and put motif details before the test statistics."""
    df = dataframe
    if df["N_COND_VARIANTS"].eq(0).all():
        df = df.drop(columns=["COND_KIND", "COND_VARIANTS", "N_REDUCED_MOTIFS"])
    df = df.drop(columns=["COND_KIND", "COND_VARIANTS", "N_COND_VARIANTS",
                          "N_REDUCED_MOTIFS", "N_EXPANDED_MOTIFS"], errors="ignore")
    df = _clean_output(df)

    first = [
        "VARIANT", "GENE", "AA_POS", "POSITIONS", "MOTIF_POSITIONS",
        "Amino_Acids", "Ref_AA", "AAcount", "AA_AF",
        "TEST", "P-value", "Statistic", "Beta", "StdErr", "DoF",
    ]
    last = ["N_used", "N_case", "N_control", "INFO"]
    first = [c for c in first if c in df]
    last = [c for c in last if c in df]
    middle = [c for c in df if c not in first + last]
    return df[first + middle + last]

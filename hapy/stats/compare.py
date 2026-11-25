"""Compare two HLA amino acid positions using BIC and conditional tests."""

import numpy as np
import pandas as pd

from .adapters.aa import AAAdapter
from .models.standard import fit_position_comparison, _validate_comparison_options
from .preprocess import prepare_y, prepare_covar, make_model_table
from .types import StandardConfig


def compare_positions(
    hladat,
    config: StandardConfig,
    position_a: str,
    position_b: str,
    *,
    famfile=None,
    covar=None,
    y=None,
    min_delta_bic=2.0,
    criterion="bic",
    conditional_threshold=0.01,
) -> pd.DataFrame:
    """Compare two amino acid positions using BIC and conditional tests.

    Parameters
    ----------
    hladat : HLAdata
        HLA data with AA genotypes loaded.
    config : StandardConfig
        Configuration with model_type="linear" or "logit".
    position_a : str
        Variant ID from hladat.AA.info["AA_ID"] for position A.
    position_b : str
        Variant ID from hladat.AA.info["AA_ID"] for position B.
    famfile : pandas.DataFrame, optional
        Sample information containing IID and PHENO when y is not supplied.
    covar : pandas.DataFrame, optional
        Covariates indexed by IID or containing an IID column.
    y : pandas.Series or array-like, optional
        Phenotype indexed by IID, or aligned to famfile row order.
        Without famfile, y must be a Series indexed by IID.
    min_delta_bic : float, optional
        Minimum BIC difference to prefer one model. Defaults to 2.
    criterion : {"bic", "conditional"}, default "bic"
        Rule used for RECOMMENDATION. Both recommendations are returned.
    conditional_threshold : float, default 0.01
        Conditional P-values below this threshold are significant.

    Returns
    -------
    pandas.DataFrame
        One row with BIC for A, B and A+B, reciprocal conditional P-values
        and degrees of freedom, sample count, both recommendations and INFO.
        RECOMMENDATION follows criterion.

    Notes
    -----
    All models use the same covariates and complete-case samples.
    Lower BIC is preferred. Conditional tests use F-tests for linear models
    and likelihood-ratio tests for logistic models.
    If the joint model cannot be fitted, its results are NaN and INFO
    describes the issue. The individual models are still compared by BIC.
    Conditional selection extends when both tests are significant, replaces A
    when only B given A is significant, and retains A when only A given B is
    significant. Neither significant, or either unavailable, is unresolved.
    """

    _validate_comparison_options(criterion, conditional_threshold)
    if not isinstance(config, StandardConfig) or config.model_type not in ("linear", "logit"):
        raise ValueError("config must be StandardConfig with model_type='linear' or 'logit'.")

    if not np.isfinite(min_delta_bic) or min_delta_bic < 0:
        raise ValueError("min_delta_bic must be finite and nonnegative.")

    if position_a == position_b:
        raise ValueError("Select two different amino acid positions.")

    if not hasattr(hladat, "AA"):
        raise ValueError("AA not loaded.")

    available = set(AAAdapter.iter_variants(hladat))
    for position in (position_a, position_b):
        if position not in available or str(position).startswith("MOTIF_"):
            raise ValueError(f"Invalid AA position: {position}")

    # Prepare phenotype and covariates
    phenotype = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, phenotype.index)
    covar_cols = list(covdf.columns) if covdf is not None else []

    # Build amino acid genotype matrices
    geno_a, _ = AAAdapter.build_geno(hladat, position_a, phenotype.index)
    geno_b, _ = AAAdapter.build_geno(hladat, position_b, phenotype.index)

    geno_a = geno_a[
        list(geno_a.attrs.get("test_columns", geno_a.columns))
    ].copy()

    geno_b = geno_b[
        list(geno_b.attrs.get("test_columns", geno_b.columns))
    ].copy()

    if geno_a.empty or geno_b.empty:
        raise ValueError("No eligible residue predictors.")

    # Rename to prevent overlapping residue column names
    geno_a.columns = [f"__A_{i}" for i in range(geno_a.shape[1])]
    geno_b.columns = [f"__B_{i}" for i in range(geno_b.shape[1])]

    a_cols = list(geno_a.columns)
    b_cols = list(geno_b.columns)

    if set(covar_cols).intersection(a_cols + b_cols + ["Y"]):
        raise ValueError("Covariates conflict with reserved columns.")

    # All models must use the same samples
    abt, _ = make_model_table(
        pd.concat([geno_a, geno_b], axis=1),
        phenotype,
        covdf,
    )

    result = fit_position_comparison(
        abt, a_cols, b_cols, covar_cols, config.model_type,
        min_delta_bic=min_delta_bic, criterion=criterion,
        conditional_threshold=conditional_threshold,
    )
    return pd.DataFrame([{
        "POSITION_A": position_a,
        "POSITION_B": position_b,
        **result,
    }])


def compare_positions_motif(hladat, config: StandardConfig, positions_a, positions_b,
                            *, famfile=None, covar=None, y=None, reference=None,
                            min_delta_bic=2.0, criterion="bic",
                            conditional_threshold=0.01) -> pd.DataFrame:
    """Compare alternative amino acid motifs using BIC and conditional tests.

    Parameters
    ----------
    hladat : HLAdata
        Retained allele mapping, or HLA allele dosages with a reference.
    config : StandardConfig
        Configuration with model_type="linear" or "logit".
    positions_a : sequence of str
        AA variant IDs for motif A, from hladat.AA.info["AA_ID"].
    positions_b : sequence of str
        AA variant IDs for motif B. Both sets must share a position and gene.
    famfile : pandas.DataFrame, optional
        Sample information containing IID and PHENO when y is not supplied.
    covar : pandas.DataFrame, optional
        Covariates indexed by IID or containing an IID column.
    y : pandas.Series or array-like, optional
        Phenotype indexed by IID, or aligned to famfile row order.
    reference : HLAReference or path, optional
        Required when no retained allele mapping is available.
    min_delta_bic : float, optional
        Minimum BIC difference to prefer one model. Defaults to 2.
    criterion : {"bic", "conditional"}, default "bic"
        Rule used for RECOMMENDATION. Both recommendations are returned.
    conditional_threshold : float, default 0.01
        Conditional P-values below this threshold are significant.

    Returns
    -------
    pandas.DataFrame
        One row with position sets, motif counts, BIC, conditional tests,
        sample count, both recommendations and INFO. RECOMMENDATION follows criterion.

    Notes
    -----
    A and B use their respective motifs. AB uses the full motif over the
    union of positions, including the shared positions. All models use
    identical samples. A+B recommends this combined motif.
    Conditional selection extends when both tests are significant, replaces A
    when only B given A is significant, and retains A when only A given B is
    significant. Neither significant, or either unavailable, is unresolved.
    """
    from ._motif import (_choose_motif_builder, _build_motifs,
                         _expanded_motif_parents, _motif_genotype)
    from .models.standard import fit_motif_comparison

    _validate_comparison_options(criterion, conditional_threshold)
    if not isinstance(config, StandardConfig) or config.model_type not in ("linear", "logit"):
        raise ValueError("config must be StandardConfig with model_type='linear' or 'logit'.")
    if not np.isfinite(min_delta_bic) or min_delta_bic < 0:
        raise ValueError("min_delta_bic must be finite and nonnegative.")
    if not hasattr(hladat, "AA"):
        raise ValueError("AA not loaded.")
    sets = []
    for positions in (positions_a, positions_b):
        if isinstance(positions, (str, bytes)):
            raise ValueError("positions must be sequences of AA variant IDs.")
        positions = list(positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("positions must be nonempty and unique.")
        sets.append(positions)
    positions_a, positions_b = sets
    if set(positions_a) == set(positions_b) or not set(positions_a).intersection(positions_b):
        raise ValueError("Select distinct position sets with at least one shared position.")
    union = list(dict.fromkeys(positions_a + positions_b))
    info = hladat.AA.info
    annotations = {}
    for variant in union:
        rows = info.loc[info["AA_ID"].eq(variant)]
        if rows.empty or str(variant).startswith("MOTIF_"):
            raise ValueError(f"Invalid AA position: {variant}")
        labels = rows[["GENE", "AA_POS"]].drop_duplicates()
        if len(labels) != 1:
            raise ValueError(f"Ambiguous AA position: {variant}")
        annotations[variant] = tuple(labels.iloc[0].astype(str))

    genes = {gene for gene, _ in annotations.values()}
    if len(genes) != 1:
        raise ValueError("Motif comparison requires positions within one gene.")
    if len(set(annotations.values())) != len(union):
        raise ValueError("Variant IDs must identify distinct AA positions.")
    gene = next(iter(genes))
    phenotype = prepare_y(famfile, y=y)
    covdf = prepare_covar(covar, phenotype.index)
    covar_cols = list(covdf) if covdf is not None else []
    builder = _choose_motif_builder(hladat, reference, phenotype.index)
    groups = [positions_a, positions_b, union]
    frames = [_build_motifs(builder, gene, [annotations[v][1] for v in group], phenotype.index)
              for group in groups]

    a, b, joint = frames
    if len(a.columns) < 2 or len(b.columns) < 2:
        raise ValueError("Selected positions have insufficient motif variation.")

    # A shared joint reference keeps both reduced models nested with partial dosages.
    parents = [_expanded_motif_parents(frame, joint) for frame in (a, b)]
    issue = ""
    if joint.empty or any(parent is None for parent in parents):
        issue = "combined motif groups do not refine both alternatives"
        for frame in frames:
            frame.attrs["reference"] = frame.sum().idxmax() if len(frame.columns) else None
    else:
        ref = joint.sum().idxmax()
        joint.attrs["reference"] = ref
        a.attrs["reference"], b.attrs["reference"] = [parent[ref] for parent in parents]
    blocks = [_motif_genotype(frame, prefix)[0]
              for frame, prefix in zip(frames, ("MOTIF_A", "MOTIF_B", "MOTIF_AB"))]
    predictors = pd.concat(blocks, axis=1)
    if set(covar_cols).intersection(["Y", *predictors.columns]):
        raise ValueError("Covariates conflict with reserved motif columns.")
    abt, _ = make_model_table(predictors, phenotype, covdf)
    if abt.empty:
        raise ValueError("No complete-case samples.")
    if not issue:
        # Check actual dosages: group labels alone cannot establish valid nesting.
        for frame, parent_map in zip((a, b), parents):
            for parent in frame:
                children = [child for child, old in parent_map.items() if old == parent]
                recovered = joint.loc[abt.index, children].sum(axis=1)
                if not np.allclose(recovered, frame.loc[abt.index, parent], rtol=1e-8, atol=1e-8):
                    issue = "combined dosages do not reconstruct both alternative motif groups"
                    break
    result = fit_motif_comparison(abt, *[list(block) for block in blocks],
                                  covar_cols, config.model_type,
                                  min_delta_bic=min_delta_bic, joint_issue=issue,
                                  criterion=criterion, conditional_threshold=conditional_threshold)
    return pd.DataFrame([{
        "POSITIONS_A": ";".join(positions_a), "POSITIONS_B": ";".join(positions_b),
        "POSITIONS_AB": ";".join(union), "GENE": gene,
        "N_MOTIFS_A": len(a.columns), "N_MOTIFS_B": len(b.columns),
        "N_MOTIFS_AB": len(joint.columns), **result,
    }])

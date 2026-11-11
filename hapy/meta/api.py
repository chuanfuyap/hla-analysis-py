"""Public API for inverse-variance and weighted Stouffer meta-analysis."""
from __future__ import annotations

import time
from ..stats.progress import format_seconds
from .io import read_input_files
from .combine import inv_var_meta_studies, stouffers_meta_studies


def meta_analyse(fileloc, data_type, beta="Beta", se="StdErr",
                 pval="P-value", variant_col="VARIANT", prefix="META", interaction=False, *,
                 test_col="TEST", n_col="N_used", case_col="N_case", control_col="N_control",
                 partner_col="Partner_col", feature_col="Variant_col", sep=","):
    """Return (inverse-variance results, weighted Stouffer results).

    Parameters
    ----------
    fileloc : str or pathlib.Path
        Text file listing one independent cohort results file per nonblank line.
    data_type : {"AA", "HLA", "SNP"}
        Genotype result type. AA standard results are classified across cohorts:
        any MULTI at a position routes its SINGLE/MULTI rows to Stouffer; only
        all-SINGLE positions use IVW. OVR rows are excluded.
    beta, se : str, defaults "Beta", "StdErr"
        Scalar coefficient and standard-error input columns for IVW. No standard
        errors are reconstructed from confidence intervals.
    pval : str, default "P-value"
        Input P-value column for Stouffer; requires 0 < P <= 1.
    variant_col : str, default "VARIANT"
        Position/variant identifier column; also names the standard output key.
    prefix : str, default "META"
        Prefix for combined statistics in both output tables.
    interaction : bool, default False
        Combine interaction results. AA inputs must be position-level omnibus
        tests; HLA/SNP pairwise effects use IVW. Output key is INTERACTION.
    test_col : str, default "TEST"
        AA standard test labels SINGLE/MULTI/OVR. Not required for HLA/SNP or
        interaction tables. Rows with missing labels are omitted.
    n_col, case_col, control_col : str or None
        Sample-count input columns, default N_used/N_case/N_control. Valid case
        and control counts give sqrt(N_eff) weights, otherwise sqrt(n_col),
        otherwise weight 1. Missing count columns are allowed; None disables one.
        Output Total N uses the selected modelling sample count.
    partner_col : str, default "Partner_col"
        Interaction partner identifier column.
    feature_col : str or None, default "Variant_col"
        Optional encoded feature column for HLA/SNP interactions. Included in
        interaction keys when present; None disables this component.
    sep : str, default ","
        Cohort-file delimiter. Use "\t" for tab-separated exports.

    Returns
    -------
    tuple[pandas.DataFrame or None, pandas.DataFrame or None]
        IVW and weighted Stouffer tables. A method with no applicable input
        returns None. A routed variant needs two contributing cohorts to appear.
        Failed/missing estimates do not contribute. Column names are explicit;
        legacy columns are not detected automatically. Effect alignment and
        independent cohorts remain the caller's responsibility.
    """
    t0 = time.perf_counter()
    inv_var, omnibus = read_input_files(
        fileloc, data_type, interaction=interaction, beta=beta, se=se, pval=pval,
        variant_col=variant_col, test_col=test_col, n_col=n_col, case_col=case_col,
        control_col=control_col, partner_col=partner_col, feature_col=feature_col, sep=sep,
    )
    identifier = "INTERACTION" if interaction else variant_col
    fe_meta, stouffer_meta = None, None
    if inv_var:
        print("Performing Inverse-Variance Meta-analysis")
        fe_meta = inv_var_meta_studies(inv_var, beta=beta, se=se, variant_col=identifier, prefix=prefix)
    if omnibus:
        print("Performing Weighted Stouffer's Method Meta-analysis")
        stouffer_meta = stouffers_meta_studies(omnibus, pval=pval, variant_col=identifier, prefix=prefix)
    print(f"Meta-analysis completed in {format_seconds(time.perf_counter() - t0)}", flush=True)
    return fe_meta, stouffer_meta

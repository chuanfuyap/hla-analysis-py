from pathlib import Path
import pandas as pd
import hapy as hp

DATA_DIR = Path(__file__).resolve().parent

def main():
    # Load example data
    phenotype = pd.read_csv(DATA_DIR / "example_pheno.txt", sep=r"\s+", index_col=0)
    covariates = pd.read_csv(DATA_DIR / "example_covar.txt", sep=r"\s+", index_col=0)

    hladat = hp.read_plink(
        DATA_DIR / "example.raw",
        format="raw",
        load_types=("AA",),
        dosage_allele="counted",
    )

    # Run logistic regression
    config = hp.stats.StandardConfig(model_type="logit", n_jobs=1)

    results = hp.stats.analyse(
        hladat,
        config,
        kind="AA",
        y=phenotype["PHENO"],
        covar=covariates,
       #condition_on="AA_DRB1_233_32548048", verbose=True,
    )

    # Save results
    results.to_csv( "aa_association.csv", index=False)
    display_cols = [c for c in ["VARIANT", "N_used", "LR_p", "Uni_p", "LRp_Unip"] if c in results]
    print(results[display_cols].to_string(index=False))
    return results


if __name__ == "__main__":
    main()
"""
SNPAdapter

Builds per-variant genotype matrices for SNPs from HLAdat.SNP.

Softcall:
- transpose allele row into sample-indexed column.

Hardcall:
- uses obt_haplo_hard to construct a sample-indexed matrix.
"""

from __future__ import annotations
import pandas as pd

from ..preprocess import subset_samples_beagle_orientation
from ._aa_haplo import obt_haplo_hard


class SNPAdapter:
    """Adapter for HLAdat.SNP genotype block."""
    KIND = "SNP"

    @staticmethod
    def iter_variants(hladat) -> list[str]:
        """
        List available SNP variants.

        Returns
        -------
        list[str]
            Unique IDs from hladat.SNP.info['AA_ID'] (legacy naming).
        """
        return list(hladat.SNP.info["AA_ID"].unique())

    @staticmethod
    def build_geno(hladat, variant_id: str, sample_index: pd.Index) -> tuple[pd.DataFrame, dict]:
        """
        Build genotype matrix for one SNP.

        Returns
        -------
        geno_df:
            IID-indexed numeric feature matrix with columns prefixed "G_snp_".
        meta:
            Dict including VARIANT and possibly POS.
        """
        df = hladat.SNP.data.copy()
        info = hladat.SNP.info.copy()

        df.index = df.index.astype(str)
        df = subset_samples_beagle_orientation(df, sample_index, hladat.type)
        df["AA_ID"] = info["AA_ID"]

        snpdf = df[df["AA_ID"] == variant_id]
        snpinfo = info[info["AA_ID"] == variant_id]

        if hladat.type == "softcall":
            geno = snpdf.drop(columns=["AA_ID"]).T.sort_index()
        else:
            geno, _AAcount, _refAA, _aalist, _ = obt_haplo_hard(snpdf)

        geno.columns = [f"G_snp_{c}" for c in geno.columns]

        meta = {
            "VARIANT": variant_id,
            "POS": snpinfo["POS"].iloc[0] if "POS" in snpinfo else None,
        }
        return geno, meta
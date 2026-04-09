"""
HLAAdapter

Builds per-variant genotype matrices for HLA alleles from HLAdat.HLA.

Softcall:
- a single allele row becomes one sample-indexed column.

Hardcall:
- uses obt_haplo_hard to construct a sample-indexed matrix.
"""

from __future__ import annotations
import pandas as pd

from ..preprocess import subset_samples_beagle_orientation
from ._aa_haplo import obt_haplo_hard


class HLAAdapter:
    """Adapter for HLAdat.HLA genotype block."""
    KIND = "HLA"

    @staticmethod
    def iter_variants(hladat) -> list[str]:
        """
        List available HLA variants.

        Returns
        -------
        list[str]
            Unique IDs from hladat.HLA.info['AA_ID'] (legacy naming).
        """
        return list(hladat.HLA.info["AA_ID"].unique())

    @staticmethod
    def build_geno(hladat, variant_id: str, sample_index: pd.Index) -> tuple[pd.DataFrame, dict]:
        """
        Build genotype matrix for one HLA variant.

        Returns
        -------
        geno_df:
            IID-indexed numeric feature matrix with columns prefixed "G_hla_".
        meta:
            Dict including VARIANT, and possibly GENE/POS if present in info.
        """
        df = hladat.HLA.data.copy()
        info = hladat.HLA.info.copy()

        df.index = df.index.astype(str)
        df = subset_samples_beagle_orientation(df, sample_index, hladat.type)
        df["AA_ID"] = info["AA_ID"]

        hladf = df[df["AA_ID"] == variant_id]
        hlainfo = info[info["AA_ID"] == variant_id]

        if hladat.type == "softcall":
            geno = hladf.drop(columns=["AA_ID"]).T.sort_index()
        else:
            geno, _AAcount, _refAA, _aalist, _ = obt_haplo_hard(hladf)

        geno.columns = [f"G_hla_{c}" for c in geno.columns]

        meta = {
            "VARIANT": variant_id,
            "GENE": hlainfo["GENE"].iloc[0] if "GENE" in hlainfo else None,
            "POS": hlainfo["POS"].iloc[0] if "POS" in hlainfo else None,
        }
        return geno, meta
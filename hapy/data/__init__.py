"""
init for `data` subpackage for hapy package.
"""

from hapy.data.io import *
from hapy.data.HLAdat import HLAdata
from hapy.data.plink import read_plink, read_genotype

__all__ = ["download_hla_inputs", "read_famfile", "read_bgl", "read_gprobs", "read_dosage", "read_plink", "read_genotype", "HLAdat", "HLAReference", "HLAAlleleMapping", "build_hla_reference", "load_hla_reference", "available_hla_genes", "map_hla_alleles", "read_aa_dosage", "aa_motif_dosages"]

from .hla_reference import HLAReference, build_hla_reference, load_hla_reference, available_hla_genes
from .hla_mapping import HLAAlleleMapping, map_hla_alleles, read_aa_dosage, aa_motif_dosages

from .hla_download import download_hla_inputs

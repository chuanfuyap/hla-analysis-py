"""
Genomics file I/O utilities.

Currently supported:
- [Phased Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

Some hardcall data processing code adapted from [HLA-TAPAS](https://github.com/immunogenomics/HLA-TAPAS/blob/master/HLAassoc/run_omnibus_test_WS.R)

Functions:
    - read_famfile: Reads PLINK FAM sample metadata file.
    - read_bgl: Reads and processes phased Beagle genotype files.
    - read_gprobs: Reads Beagle probability files and transforms to dosage.
    - read_dosage: Reads dosage files and processes for HLAdat.
    - breakitup: Utility to parse variant IDs into components.
    - getSampleIDs: Extracts sample IDs from a phased genotype file.
"""

__all__ = ["read_famfile", "read_bgl", "read_gprobs", "read_dosage"]

import abc
from typing import Optional, List, Tuple, Any
import pandas as pd
import numpy as np
import gc
import time

from hapy.data.HLAdat import HLAdata

# --- Utility Functions ---

def breakitup(variantID: str):
    """
    Decompose variant IDs into structured components for easier sorting and analysis.

    Parameters
    ----------
    variantID : str
        Variant ID from genotype files.

    Returns
    -------
    Tuple[str, str, str, str, str]
        idname: Cleaned up ID.
        variantype: Variant type (SNP, HLA, AA, etc.).
        genename: Name of gene if amino acid variant.
        aapos: Amino acid position number (or NaN).
        genepos: Genomic coordinate (or NaN).
    """
    tmpID = variantID.replace("*", "_").replace("SNPS_", "SNP_")
    tokens = tmpID.split("_")
    if len(tokens) > 1:
        idname = ("_").join(tokens[:4])
        while len(tokens) < 4:
            tokens.append(np.nan)
        variantype, genename, aapos, genepos = tokens[0], tokens[1], tokens[2], tokens[3]
    else:
        idname = variantID
        variantype = "SNP" if variantID.startswith("rs") else variantID
        genename = aapos = genepos = np.nan
    return idname, variantype, genename, aapos, genepos

def getSampleIDs(phasedfileloc: str) -> List[str]:
    """
    Extracts sample IDs from the header of a phased genotype file.

    Parameters
    ----------
    phasedfileloc : str
        File path to phased genotype file (e.g. Beagle .bgl).

    Returns
    -------
    List[str]
        List of sample IDs corresponding to individuals in the dataset.
    """
    with open(phasedfileloc, "r") as f:  #pylint: disable=W1514
        sampIDs = np.array(f.readline().split()[2:])
        idCount2 = len(sampIDs)
        ix = [i for i in range(1, idCount2, 2)]
    return list(sampIDs[ix])

def _apply_r2_filter(df: pd.DataFrame, filter_R2: Optional[str], R2_minimum: float) -> pd.DataFrame:
    """
    Filters variants in a DataFrame based on imputation R2 quality threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Variant matrix to filter.
    filter_R2 : Optional[str]
        File path to .r2 file for variant imputation quality.
    R2_minimum : float
        Minimum R2 threshold for filtering.

    Returns
    -------
    pd.DataFrame
        Filtered variant dataframe.
    """
    if filter_R2:
        r2 = pd.read_csv(filter_R2, sep=r"\s+", header=None, index_col=0)
        safe = r2[r2[1] > R2_minimum].index
        df = df.loc[safe]
    return df

def _add_variant_info(df: pd.DataFrame, index_name: str = "SNP") -> pd.DataFrame:
    """
    Adds variant metadata columns to the input DataFrame using variant IDs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing variant data, indexed by variant ID.
    index_name : str, optional
        Name to assign to variant index column.

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata columns ['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS'].
    """
    df.index.name = index_name
    df["SNP"] = df.index
    df[['AA_ID', 'TYPE', 'GENE', 'AA_POS', 'POS']] = df.apply(lambda x: breakitup(x["SNP"]), axis=1, result_type="expand")
    return df.drop(columns=["SNP"], axis=1)

# --- Abstract Base Reader ---

class GenomicsFileReader(abc.ABC):
    """
    Abstract base class for all genomics file readers.

    Parameters
    ----------
    filter_R2 : Optional[str], default None
        File path for imputation quality filtering.
    R2_minimum : float, default 0.5
        R2 threshold for variant imputation quality.
    simpleQC : bool, default True
        Whether to perform simple minor allele frequency filtering.
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load for downstream processing.
    """
    def __init__(self, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')):
        self.filter_R2 = filter_R2
        self.R2_minimum = R2_minimum
        self.simpleQC = simpleQC
        if isinstance(load_types, str):
            load_types = (load_types,)
        self.load_types = load_types

    @abc.abstractmethod
    def read(self) -> Any:
        """
        Abstract method to read and process the file.
        Must be implemented by subclasses.

        Returns
        -------
        Any
            Processed data object (typically HLAdat or DataFrame).
        """
        pass  #pylint: disable=W0107

    def postprocess(self, hladat: HLAdata):
        """
        Apply post-processing steps (e.g. MAF filter, print summary).

        Parameters
        ----------
        hladat : HLAdata
            Data object containing processed genomics data.

        Returns
        -------
        HLAdata
            Post-processed HLAdata object.
        """
        if self.simpleQC:
            print("----------------------------------------------------", flush=True)
            print("PERFORMING SIMPLE MAF FILTER: droppping 0.5% allele frequency", flush=True)
            print("----------------------------------------------------", flush=True)
            hladat.maf_filter()

        print("---------------------", flush=True)
        # List of expected possible types
        possible_types = self.load_types if hasattr(self, "load_types") else ["SNP", "HLA", "AA"]
        # Find the first available attribute in hladat for sample size
        sample_obj = None
        for lt in possible_types:
            if hasattr(hladat, lt):
                sample_obj = getattr(hladat, lt)
                break

        if sample_obj is not None:
            if hasattr(hladat, "type") and hladat.type == "hardcall":
                sample_size = len(sample_obj.data.columns) // 2
            else:
                sample_size = len(sample_obj.data.columns)
            print(f"Sample Size:\t {sample_size}", flush=True)
        else:
            print("No valid data type found for sample size.", flush=True)

        # Print counts only for attributes that exist
        for load_type in possible_types:
            if hasattr(hladat, load_type):
                obj = getattr(hladat, load_type)
                count = obj.info.AA_ID.nunique()
                print(f"Number of {load_type} variants:\t{count}", flush=True)

        print("---------------------", flush=True)
        gc.collect()
        return hladat
    
    def _load_and_process(self, file_path, file_read_fn, call_type, variant_index_name="SNP", convert_dosage=False, drop_alleles=False, extra_args=None):
        start = time.time()
        print("----------------", flush=True)
        print("READING IN DATA", flush=True)
        print(f"File:\t{file_path}", flush=True)
        print("----------------", flush=True)
        df = file_read_fn(file_path, extra_args)
        print(f"Elapsed time for loading: {time.time() - start:.4f} seconds", flush=True)
        start2 = time.time()
        df = _apply_r2_filter(df, self.filter_R2, self.R2_minimum)
        df = _add_variant_info(df, index_name=variant_index_name)
        hladat = HLAdata(df, call_type, load_types=self.load_types)
        if convert_dosage:
            print("---------------------", flush=True)
            print("CONVERTING TO DOSAGE", flush=True)
            print("---------------------", flush=True)
            hladat.convert_dosage()
        if drop_alleles:
            for load_type in self.load_types:
                if hasattr(hladat, load_type):
                    obj = getattr(hladat, load_type)
                    obj.data.drop(columns=["alleleA", "alleleB"], axis=1, inplace=True)
        del df
        hladat = self.postprocess(hladat)
        print(f"Elapsed time for processing: {time.time() - start2:.4f} seconds", flush=True)
        return hladat

# --- Specific Readers ---

class FamFileReader(GenomicsFileReader):
    """
    Reader for PLINK FAM files.

    Parameters
    ----------
    fileloc : str
        File path to .fam file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['FID', 'IID', 'FAT', 'MOT', 'SEX', 'PHENO'].
    """
    def __init__(self, fileloc: str):
        super().__init__()
        self.fileloc = fileloc

    def read(self) -> pd.DataFrame:
        fam = pd.read_csv(
            self.fileloc,
            sep=r"\s+",
            names=["FID", "IID", "FAT", "MOT", "SEX", "PHENO"],
            na_values=[-9, "-9"]
        )
        return fam

class BGLFileReader(GenomicsFileReader):
    """
    Reader for phased Beagle genotype files.

    Parameters
    ----------
    fileloc : str
        File path to Beagle file (.bgl).
    filter_R2 : Optional[str], default None
        File path for R2 filtering.
    R2_minimum : float, default 0.5
        R2 threshold.
    simpleQC : bool, default True
        Whether to apply MAF filter.
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load.

    Returns
    -------
    HLAdata
        HLAdat object containing processed genotype data.
    """
    def __init__(self, fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')):
        super().__init__(filter_R2, R2_minimum, simpleQC, load_types)
        self.fileloc = fileloc

    def read(self) -> HLAdata:
        def file_read_fn(file_path, _):
            df = pd.read_csv(file_path, sep=r"\s+", header=0, index_col=1)
            marker_col = df.columns[0]
            df = df[df[marker_col] == "M"].drop(marker_col, axis=1)
            return df

        return self._load_and_process(self.fileloc, file_read_fn, "hardcall")

class GProbsFileReader(GenomicsFileReader):
    """
    Reader for Beagle genotype probability files.

    Parameters
    ----------
    fileloc : str
        File path to Beagle probability file.
    filter_R2 : Optional[str], default None
        File path for R2 filtering.
    R2_minimum : float, default 0.5
        R2 threshold.
    simpleQC : bool, default True
        Whether to apply MAF filter.
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load.

    Returns
    -------
    HLAdata
        HLAdat object containing processed genotype probability data.
    """
    def __init__(self, fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')):
        super().__init__(filter_R2, R2_minimum, simpleQC, load_types)
        self.fileloc = fileloc

    def read(self) -> HLAdata:
        def file_read_fn(file_path, _):
            df = pd.read_csv(file_path, sep=r"\s+", header=0, index_col=0)
            assert df.index.name == "marker", (
                "ERROR: File appears to be modified. If this is a SNP2HLA output, please use the dosage output with `read_dosage(dosagefileloc, phasedfileloc)` instead."
            )
            return df

        return self._load_and_process(self.fileloc, file_read_fn, "softcall", convert_dosage=True)

class DosageFileReader(GenomicsFileReader):
    """
    Reader for SNP2HLA dosage files.

    Parameters
    ----------
    dosagefileloc : str
        File path to dosage file.
    phasedfileloc : str
        File path to phased genotype file (for sample IDs).
    filter_R2 : Optional[str], default None
        File path for R2 filtering.
    R2_minimum : float, default 0.5
        R2 threshold.
    simpleQC : bool, default True
        Whether to apply MAF filter.
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load.

    Returns
    -------
    HLAdata
        HLAdat object containing processed dosage data.
    """
    def __init__(self, dosagefileloc: str, phasedfileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5, simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')):
        super().__init__(filter_R2, R2_minimum, simpleQC, load_types)
        self.dosagefileloc = dosagefileloc
        self.phasedfileloc = phasedfileloc

    def read(self) -> HLAdata:
        def file_read_fn(file_path, extra_args):
            sampleIDs = getSampleIDs(extra_args)
            header = ["alleleA", "alleleB"] + sampleIDs
            df = pd.read_csv(file_path, sep=r"\s+", header=None, index_col=0)
            df.columns = header
            return df

        return self._load_and_process(self.dosagefileloc, file_read_fn, "softcall", drop_alleles=True, extra_args=self.phasedfileloc)

# --- Facade Functions ---

def read_famfile(fileloc: str) -> pd.DataFrame:
    """
    Reads PLINK FAM file and gives it appropriate headers.

    Parameters
    ----------
    fileloc : str
        File location of the FAM file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['FID', 'IID', 'FAT', 'MOT', 'SEX', 'PHENO'].
    """
    return FamFileReader(fileloc).read()

def read_bgl(fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5,
             simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')) -> HLAdata:
    """
    Processes phased Beagle file and stores it as HLAdat object.

    Parameters
    ----------
    fileloc : str
        File location of the Beagle (phased) file.
    filter_R2 : Optional[str], default None
        File location of the bgl.r2 file for filtering variants with low R2 values.
    R2_minimum : float, default 0.5
        Minimum R2 value for variant filtering.
    simpleQC : bool, default True
        If True, performs simple MAF filter (drops variants with allele frequency below 0.5%).
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load ('HLA', 'SNP', 'AA').

    Returns
    -------
    HLAdata
        HLAdat object with genomic data files as dataframes.
    """
    return BGLFileReader(fileloc, filter_R2, R2_minimum, simpleQC, load_types).read()

def read_gprobs(fileloc: str, filter_R2: Optional[str] = None, R2_minimum: float = 0.5,
                simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')) -> HLAdata:
    """
    Processes Beagle probability file, transforms it into dosage file and stores as HLAdat object.

    Parameters
    ----------
    fileloc : str
        File location of the Beagle probability file.
    filter_R2 : Optional[str], default None
        File location of the bgl.r2 file for filtering variants with low R2 values.
    R2_minimum : float, default 0.5
        Minimum R2 value for variant filtering.
    simpleQC : bool, default True
        If True, performs simple MAF filter (drops variants with allele frequency below 0.5%).
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load.

    Returns
    -------
    HLAdata
        HLAdat object with genomic probability data as dataframes.
    """
    return GProbsFileReader(fileloc, filter_R2, R2_minimum, simpleQC, load_types).read()

def read_dosage(dosagefileloc: str, phasedfileloc: str, filter_R2: Optional[str] = None,
                R2_minimum: float = 0.5, simpleQC: bool = True, load_types: Tuple[str, ...] = ('HLA', 'SNP', 'AA')) -> HLAdata:
    """
    Processes dosage file and stores it as HLAdat object.

    Parameters
    ----------
    dosagefileloc : str
        File location of the dosage file.
    phasedfileloc : str
        File location of the phase file (for sample IDs).
    filter_R2 : Optional[str], default None
        File location of the bgl.r2 file for filtering variants with low R2 values.
    R2_minimum : float, default 0.5
        Minimum R2 value for variant filtering.
    simpleQC : bool, default True
        If True, performs simple MAF filter (drops variants with allele frequency below 0.5%).
    load_types : Tuple[str, ...], default ('HLA', 'SNP', 'AA')
        Types of data to load.

    Returns
    -------
    HLAdata
        HLAdat object with dosage data as dataframes.
    """
    return DosageFileReader(dosagefileloc, phasedfileloc, filter_R2, R2_minimum, simpleQC, load_types).read()
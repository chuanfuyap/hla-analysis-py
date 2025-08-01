"""
Function to load and modify HLA dataset.

Currently support:
- [Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

If you have VCF file, please use Beagle's util files for converting to bgl/gprob files.
"""
from types import SimpleNamespace
from typing import Dict, Any
import pandas as pd

class HLAdata:
    """
    The object type to store the all genotype information for HLA, which includes HLA-4digit (HLA), Amino Acid (AA) and SNPs.
    """
    def __init__(self, genomedata: pd.DataFrame, data_type: str, load_types: tuple=('HLA', 'SNP', 'AA')):
        """
        Initialize HLAdata object with formatted genome data.

        Parameters
        ------------
        genomedata: pd.DataFrame
            dataframe formatted with additional info when reading in the data.
        data_type : str
            'hardcall' or 'softcall' (dosage).
        load_types : tuple
            tuple of strings indicating which types of data to load. Options are 'HLA', 'SNP', and 'AA'.

        The class will create one attribute each for HLA, SNP, and AA data, which are SimpleNamespace objects containing 'data' and 'info' attributes.
        
        The 'data' attribute contains the genotype count/dosage data, while the 'info' attribute contains additional information such as allele IDs, gene names, amino acid positions, and genomic positions.

        - 'hardcall' count data stores data as A/T/C/G depending on haplotype or imputation method
        - 'softcall' dosage data stores data as probabilities of the alleles.

        Example:
        ```python
        hladat.AA.data  # Pandas DataFrame with AA genotype counts/dosage
        hladat.AA.info  # Pandas DataFrame with additional information about AA alleles (AA_ID,	GENE,	AA_POS,	POS)
        ```
        """
        if not isinstance(genomedata, pd.DataFrame):
            raise TypeError("genomedata must be a pandas DataFrame")
        if not isinstance(data_type, str) or data_type not in ["hardcall", "softcall"]:
            raise ValueError("data_type must be 'hardcall' or 'softcall'")

        self.type = data_type
        if 'HLA' in load_types:
            self.HLA = self._add_data(genomedata[genomedata.TYPE == "HLA"], "HLA")
        if 'SNP' in load_types:
            self.SNP = self._add_data(genomedata[genomedata.TYPE == "SNP"], "SNP")
        if 'AA' in load_types:
            self.AA = self._add_data(genomedata[genomedata.TYPE == "AA"], "AA")

        if data_type == "softcall":
            alleleAB = genomedata[["alleleA", "alleleB", "TYPE"]]
            for name in ["HLA", "SNP", "AA"]:
                obj = getattr(self, name, None)  # Get attribute if it exists, else None
                if obj is not None:
                    mask = alleleAB.TYPE == name
                    # .loc to ensure index alignment
                    obj.info[["alleleA", "alleleB"]] = alleleAB.loc[mask, ["alleleA", "alleleB"]].values

    def _add_data(self, genomedata: pd.DataFrame, allele_type: str) -> SimpleNamespace:
        """
        Formats and returns genotype data and info as a SimpleNamespace.
        """
        data = genomedata.drop(columns=["AA_ID","TYPE", "GENE", "AA_POS", "POS"], axis=1).copy()
        info = genomedata[["AA_ID", "GENE", "AA_POS", "POS"]].copy()

        datadict: Dict[str, Any] = {"info": info}

        if allele_type == "SNP":
            ### this check is done for older version of HLA imputation which doesn't give AA position.
            ### this would move AA_POS to POS.
            tmp = info.AA_POS.fillna(0).astype("int")

            if tmp.max()>33333: ## hypothetical max range of amino acid positions
                tmp_pos = info.loc[:,"POS"].values
                info.loc[:,"POS"] = info.loc[:,"AA_POS"].values
                info.loc[:,"AA_POS"] = tmp_pos

            extrasnps = [x for x in data.index if x.startswith("SNP")] ## EXTRA DATA FOR HLA GENE SNPS THAT USES A/T AS WELL.
            datadict["extradata"] = data.loc[extrasnps].copy()

            normalsnps = [x for x in data.index if not x.startswith("SNP")]
            datadict["data"] = data.loc[normalsnps].copy()

        elif allele_type == "AA":
            datadict["data"] = data.copy()

        else: ## else HLA
            data.index = [ix.replace("*","_").replace(":","") for ix in data.index]
            info.index = [ix.replace("*","_").replace(":","") for ix in info.index]

            datadict["data"] = data.copy()

        return SimpleNamespace(**datadict)

    def maf_filter(self, allele_filter: float = 0.005) -> None:
        """
        Goes through all HLA allele (4digit/SNP/AA) and perform quality control, removing variants with sample frequency <0.5% (Default)

        allele_filter: float
            allele frequency (% of alleles across all samples)
        """

        if self.type == "hardcall":
            for name in ["HLA", "SNP", "AA"]:
                obj = getattr(self, name, None)  # Get attribute if it exists, else None
                if obj is not None:
                    obj.data = _qc_hard(obj.data, allele_filter)
                    obj.info = obj.info.loc[obj.data.index]

        elif self.type == "softcall":
            for name in ["HLA", "SNP", "AA"]:
                obj = getattr(self, name, None)  # Get attribute if it exists, else None
                if obj is not None:
                    obj.data = _qc_prob(obj.data, allele_filter)
                    obj.info = obj.info.loc[obj.data.index]

        else:
            raise ValueError(f"Unknown data type '{self.type}'.")

    def convert_dosage(self):
        """
        Converts probability/likelihood data into dosage information where 2(AA) + 1(AB) + 0(BB)

        This is only executed if the data type is 'softcall'.
        """
        if self.type != "softcall":
            raise ValueError("convert_dosage can only be used with 'softcall' data type.")

        for name in ["HLA", "SNP", "AA"]:
            obj = getattr(self, name, None)  # Get attribute if it exists, else None
            if obj is not None:
                if name == "AA":
                    obj.data = _makedosage_AA(obj.data)
                else:
                    obj.data = _makedosage_HLA_SNP(obj.data)

def _qc_hard(dataframe: pd.DataFrame, allele_filter: float) -> pd.DataFrame:
    """
    Takes hardcall data and perform maf filter
    """
    df = dataframe.copy()

    ### QC allele frequency
    # this counts the A/T/C/G and normalize true gives proportions, so the min would be MAF
    maf = df.apply(lambda x: x.value_counts(normalize=True).values.min(), axis=1)
    df = df.loc[maf>allele_filter] ## keep only those above filter limit
    df = df.loc[maf!=1] ## drop those which has allele frequency of 100%

    return df

def _qc_prob(dataframe: pd.DataFrame, allele_filter: float) -> pd.DataFrame:
    """
    Takes softcall/dosage data and perform maf filter

    Parameters
    -----------
    dataframe:
        Beagle dataset to undergo quality control
    allele_filter: float
        allele frequency (% of alleles across all samples)
    Returns
    ------------
    df: Pandas DataFrame
        processed dataset

    """
    df = dataframe.copy().fillna(0)

    ### QC allele frequency
    sample_count = df.shape[1] * 2 # diploid
    highfreqallele = df.sum(axis=1)/sample_count > allele_filter
    df = df.loc[highfreqallele]
    ## this reverse is needed since there is no MAF, it would just be dose of single allele's dose reported, this would remove cases where it is dominated by only one allele
    highfreqallele_rev = df.sum(axis=1)/sample_count < (1-allele_filter)
    df = df.loc[highfreqallele_rev]

    return df

def dosage_AA(x: pd.DataFrame) -> float:
    """
    Dosage computation function FOR AA

    Parameters
    ----------
    x: pd.Series
    """
    AA, AB, BB, check = x.iloc[0], x.iloc[1], x.iloc[2], x.iloc[3]

    if check==1: # this is for Michigan HLA imputation output format where it uses A for absence and T for presence, and is of reverse order in alleleA/B.
        dose = (0*AA)+ (1*AB) + (2*BB) # because in new AT, allele B is T for present.
    else:
        dose = (2*AA)+ (1*AB) + (0*BB) # makes allele A the effect allele and allle B the reference.
    return dose

def dosage_HLA_SNP(x: pd.DataFrame, newAT: bool) -> float:
    """
    Dosage computation function FOR SNP/HLA
    
    Parameters
    ----------
    x: pd.Series

    newAT: boolean
        this is for Michigan HLA imputation output format where it uses A for absence and T for presence, and is of reverse order in alleleA/B.
    """
    AA,AB,BB = x.iloc[0],x.iloc[1],x.iloc[2]

    if newAT:
        dose = (0*AA)+ (1*AB) + (2*BB) # because in new AT, allele B is T for present.
    else:
        dose = (2*AA)+ (1*AB) + (0*BB) # makes allele A the effect allele and allle B the reference.
    return dose

def _makedosage_HLA_SNP(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converts genotype probabilities to dosages, used for HLA/SNP. 
    
    Parameters
    -----------
    dataframe: pd.DataFrame
        dataframe with
        - index of variant markers
        - column: ["alleleA", "alleleB", and sample IDs] and 
    """
    at_checks = _checkAT_HLA_SNP(dataframe["alleleA"],dataframe["alleleB"])

    samplenames = dataframe.columns[2:]

    df = dataframe[samplenames].T.copy()
    df["samples"] = df.index
    df.samples = df.samples.apply(lambda x : x.split('.')[0])

    dosage_dict = {}
    for samp in df.samples.unique():
        sampdf = df[df.samples==samp]
        dose = dosage_HLA_SNP(sampdf,at_checks)
        dosage_dict[samp]=dose

    dosedf = pd.DataFrame(dosage_dict)
    dosedf = dosedf.drop("samples", axis=0)

    return dosedf

def _makedosage_AA(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converts genotype probabilities to dosages, used for AA. 
    
    Parameters
    -----------
    dataframe: pd.DataFrame
        dataframe with
        - index of variant markers (AA_ID)
        - column: ["alleleA", "alleleB", and sample IDs] and 
    """
    # Extract sample names (assumes first two columns are alleleA and alleleB)
    samplenames = dataframe.columns[2:]
    absamp = dataframe[["alleleA", "alleleB"]]
    # Check for Michigan A/T format
    at_checks = absamp.apply(_checkAT_AA, axis=1)

    # Prepare result dictionary
    dosage_dict = {}
    # Create DataFrame with samples as rows, 
    df = dataframe[samplenames].T.copy()
    df["samples"] = df.index
    df.samples = df.samples.apply(lambda x : x.split('.')[0])

    for samp in df.samples.unique():
        sampdf = df[df.samples==samp]
        sampdf = pd.concat([sampdf, at_checks], ignore_index=True)
        dose = sampdf.apply(dosage_AA, axis=0)
        dosage_dict[samp]=dose

    # Combine all dosages into final DataFrame
    dosedf = pd.DataFrame(dosage_dict)
    dosedf = dosedf.drop("samples", axis=0)

    return dosedf

def _checkAT_HLA_SNP(colA, colB):
    """
    Check if using Michigan HLA imputation A/T absence/presence format. FOR SNP/HLA.
    """
    return set(colA.unique()) == {"A"} and set(colB.unique()) == {"T"}

def _checkAT_AA(df):
    """
    Check if using Michigan HLA imputation A/T absence/presence format. FOR AA.
    """    
    name_check=df.name
    return df.get("alleleA") == "A" and df.get("alleleB") == "T" and str(name_check).startswith("AA_")
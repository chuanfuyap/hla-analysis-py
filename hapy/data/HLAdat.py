"""
Function to load and modify HLA dataset.

Currently support:
- [Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

If you have VCF file, please use Beagle's util files for converting to bgl/gprob files.
"""
import numpy as np

class HLAdata:
    """
    The object type to store the all genotype information for HLA.
    """
    def __init__(self, genomedata, data_type):
        """
        Creates HLAdata object given formatted genome data

        Parameters
        ------------
        genomedata: formatted Pandas DataFrame,
            dataframe formatted with additional info when reading in the data.
        type: str
            data type, hardcall or softcall (dosage) file.
        """

        self.HLA = self.add_data(genomedata[genomedata.TYPE=="HLA"], "HLA")
        self.SNP = self.add_data(genomedata[genomedata.TYPE=="SNP"], "SNP")
        self.AA = self.add_data(genomedata[genomedata.TYPE=="AA"], "AA")
        self.type = data_type # hardcall or probability/dosage

    def add_data(self, genomedata, allele_type):
        """
        Takes respective data (HLA/SNP/AA) and add it to HLAdata object

        Parameters
        ------------
        genomedata: formatted Pandas DataFrame,
            dataframe formatted with additional info when reading in the data.
        Returns
        ------------
        datadict: dictionary
            dictionary containing the "data" and "info", which are genotype data and their information respectively.
        """
        datadict = {}
        data = genomedata.drop(columns=["AA_ID","TYPE", "GENE", "AA_POS", "POS"], axis=1).copy()
        info = genomedata[["AA_ID", "GENE", "AA_POS", "POS"]].copy()

        if allele_type == "SNP":
            ### this check is done for older version of HLA imputation which doesn't give AA position.
            ### this would move AA_POS to POS.
            tmp = info.copy()
            tmp = tmp.AA_POS.fillna(0).astype("int")

            if tmp.max()>33333: ## hypothetical max range of amino acid positions
                tmp_pos = info.loc[:,"POS"].values
                info.loc[:,"POS"] = info.loc[:,"AA_POS"].values
                info.loc[:,"AA_POS"] = tmp_pos
            datadict["info"] = info
            datadict["data"] = data
        elif allele_type == "AA":
            ### now sectioning just a amino acids with >1 amino acids at the same position
            tmpix = np.where(info.groupby("AA_ID").count()["POS"]>1)[0]
            tmpix = info.groupby("AA_ID").count().index[tmpix]
            
            info = info[info.AA_ID.isin(tmpix)]
            data = data.loc[info.index]
            datadict["info"] = info
            datadict["data"] = data

        else: ## else HLA/AA
            datadict["info"] = info
            datadict["data"] = data

        return datadict

    def qualitycontrol(self, allelefilter=0.01):
        """
        Goes through all HLA allele (4digit/SNP/AA) and perform quality control, removing variants with sample frequency <1% (Default)

        allelfilter: float
            allele frequency (% of alleles across all samples)
        """

        if self.type == "hardcall":
            self.SNP["data"] = self.qcSNP_hard(self.SNP["data"], allelefilter)
            self.SNP["info"] =self.SNP["info"].loc[self.SNP["data"].index]
            self.HLA["data"] = self.qcHLA_hard(self.HLA["data"], allelefilter)
            self.HLA["info"] =self.HLA["info"].loc[self.HLA["data"].index]
            self.AA["data"] = self.qcAA_hard(self.AA["data"], allelefilter)
            self.AA["info"] =self.AA["info"].loc[self.AA["data"].index]
        elif self.type == "softcall":
            #self.SNP["data"] = self.qcSNP_soft(self.SNP["data"], allelefilter)
            #self.HLA["data"] = self.qcHLA_soft(self.HLA["data"], allelefilter)
            #self.AA["data"] = self.qcAA_soft(self.AA["data"], allelefilter)
            pass
        else:
            print("wrong data type set, please investigate")

    def qcSNP_hard(self, dataframe, allelefilter):
        """
        Performs QC on SNPs found in HLA.
        """
        df = dataframe.copy()

        ### QC allele frequency
        maf = df.apply(lambda x: x.value_counts(normalize=True).values.min(), axis=1)
        df = df.loc[maf>allelefilter]

        return df

    def qcHLA_hard(self, dataframe, allelefilter):
        """
        Performs QC on classic HLA 4-Digit alleles
        """
        df = dataframe.copy()

        ### this is done since some HLA imputation use P for presence while newer one uses T.
        for col in df.columns:
            df[col] = df[col].map({"P":"T", "A":"A", "T":"T"})

        ### QC allele frequency
        highfreqallele = (df=="T").sum(1)/df.shape[1] > allelefilter
        df = df.loc[highfreqallele]

        return df

    def qcAA_hard(self, dataframe, allelefilter):
        """
        Performs QC on AA found in HLA.
        """
        df = dataframe.copy()

        ### QC allele frequency
        keepallele = df.apply(lambda x: filter_allele(x, allelefilter), axis=1)
        df = df.loc[keepallele]

        return df

def filter_allele(series, allelefilter):
    """
    takes series and determine if allele should be kept based on what is present and frequency.
    """
    allelecount = series.value_counts(normalize=True)
    if allelecount.index.shape[0]==1:
        if allelecount.index[0]=="A" and allelecount.values==1:
            return False
        else:
            return True
    elif allelecount.index.shape[0]==2:
        ## checks now whether if it is A/P, A/T or something else
        aminoacids = set(allelecount.index)
        if aminoacids == {"A", "P"} or aminoacids == {"A", "T"}:
            if (series=="A").sum()/series.shape[0] > (1-allelefilter):
                ## this check is basically if there is e.g. 99% absence in this allele.
                return False
            else:
                return True
        else:
        ## means now there is something that isn't usual A/P
            if series.value_counts(normalize=True).values.min() < allelefilter:
                return False
            else:
                return True
    else:
        ## for possible error checking
        return "ERROR"

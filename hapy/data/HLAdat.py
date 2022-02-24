"""
Function to load and modify HLA dataset.

Currently support:
- [Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

If you have VCF file, please use Beagle's util files for converting to bgl/gprob files.
"""
from types import SimpleNamespace

import pandas as pd

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
        if data_type=="hardcall":
            self.HLA = self.add_data(genomedata[genomedata.TYPE=="HLA"], "HLA")
            self.SNP = self.add_data(genomedata[genomedata.TYPE=="SNP"], "SNP")
            self.AA = self.add_data(genomedata[genomedata.TYPE=="AA"], "AA")
            self.type = data_type # hardcall or probability/dosage
        elif data_type=="softcall":
            alleleAB = genomedata[["alleleA", "alleleB", "TYPE"]]
            self.HLA = self.add_data(genomedata[genomedata.TYPE=="HLA"], "HLA")
            self.HLA.info[["alleleA", "alleleB"]] = alleleAB[alleleAB.TYPE=="HLA"][["alleleA", "alleleB"]]

            self.SNP = self.add_data(genomedata[genomedata.TYPE=="SNP"], "SNP")
            self.SNP.info[["alleleA", "alleleB"]] = alleleAB[alleleAB.TYPE=="SNP"][["alleleA", "alleleB"]]

            self.AA = self.add_data(genomedata[genomedata.TYPE=="AA"], "AA")
            self.AA.info[["alleleA", "alleleB"]] = alleleAB[alleleAB.TYPE=="AA"][["alleleA", "alleleB"]]

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

            extrasnps = [x for x in data.index if x.startswith("SNP")] ## EXTRA DATA FOR HLA GENE SNPS THAT USES A/T AS WELL.

            extradata = data.loc[extrasnps].copy()
            datadict["extradata"] = extradata

            normalsnps = [x for x in data.index if not x.startswith("SNP")]
            data = data.loc[normalsnps]
            datadict["data"] = data
            outputdict = SimpleNamespace(**datadict)
        elif allele_type == "AA":
            ### now sectioning just a amino acids with >1 amino acids at the same position
            #tmpix = np.where(info.groupby("AA_ID").count()["POS"]>1)[0]
            #tmpix = info.groupby("AA_ID").count().index[tmpix]

            #info = info[info.AA_ID.isin(tmpix)]
            #data = data.loc[info.index]
            extradata = None
            datadict["info"] = info
            datadict["data"] = data

            outputdict = SimpleNamespace(**datadict)
        else: ## else HLA/AA
            data.index = [ix.replace("*","_").replace(":","") for ix in data.index]
            info.index = [ix.replace("*","_").replace(":","") for ix in info.index]
            extradata = None
            datadict["info"] = info
            datadict["data"] = data
            outputdict = SimpleNamespace(**datadict)

        return outputdict

    def qualitycontrol(self, allelefilter=0.01):
        """
        Goes through all HLA allele (4digit/SNP/AA) and perform quality control, removing variants with sample frequency <1% (Default)

        allelfilter: float
            allele frequency (% of alleles across all samples)
        """

        if self.type == "hardcall":
            self.SNP.data = self.qcSNP_hard(self.SNP.data, allelefilter)
            self.SNP.info =self.SNP.info.loc[self.SNP.data.index]

            self.HLA.data = self.qcHLA_hard(self.HLA.data, allelefilter)
            self.HLA.info =self.HLA.info.loc[self.HLA.data.index]

            self.AA.data = self.qcAA_hard(self.AA.data, allelefilter)
            self.AA.info =self.AA.info.loc[self.AA.data.index]

        elif self.type == "softcall":
            self.SNP.data = self.qc_prob(self.SNP.data, allelefilter)
            self.SNP.info =self.SNP.info.loc[self.SNP.data.index]

            self.HLA.data = self.qc_prob(self.HLA.data, allelefilter)
            self.HLA.info =self.HLA.info.loc[self.HLA.data.index]

            self.AA.data = self.qc_prob(self.AA.data, allelefilter)
            self.AA.info =self.AA.info.loc[self.AA.data.index]

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
        df = df.loc[maf<(1-allelefilter)]

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
        highfreqallele2 = (df=="T").sum(1)/df.shape[1] < (1-allelefilter)
        df = df.loc[highfreqallele2]

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

    def qc_prob(self, dataframe, alellefilter=0.01):
        """
        Takes dosage data and perform quality control, variants with sample frequency < 1%

        Parameters
        -----------
        dataframe:
            Beagle dataset to undergo quality control
        allelfilter: float
            allele frequency (% of alleles across all samples)
        Returns
        ------------
        df: Pandas DataFrame
            processed dataset

        """
        df = dataframe.copy().fillna(0)

        ### QC allele frequency

        highfreqallele = df.sum(1)/(df.shape[1]*2) > alellefilter
        df = df.loc[highfreqallele]

        highfreqallele2 = df.sum(1)/(df.shape[1]*2) < (1-alellefilter)
        df = df.loc[highfreqallele2]

        return df

    def convertDosage(self):
        """
        Converts probability/likelihood data into dosage information where 2(AA) + 1(AB) + 0(BB)
        """
        self.SNP.data = makedosage_(self.SNP.data)
        self.HLA.data = makedosage_(self.HLA.data)
        self.AA.data = makedosage(self.AA.data)

def dosage(x):
    """
    Dosage computation function FOR AA

    Parameters
    ----------
    newAT: boolean
        this is for Michigan HLA imputation output format where it uses A for absence and T for presence, and is of reverse order in alleleA/B.
    """
    AA,AB,BB,check = x.iloc[0],x.iloc[1],x.iloc[2],x.iloc[3]

    if check==1:
        dose = (0*AA)+ (1*AB) + (2*BB) # because in new AT, allele B is T for present.
    else:
        dose = (2*AA)+ (1*AB) + (0*BB) # makes allele A the effect allele and allle B the reference.
    return dose
def dosage_(x, newAT):
    """
    Dosage computation function FOR SNP/HLA
    Parameters
    ----------
    newAT: boolean
        this is for Michigan HLA imputation output format where it uses A for absence and T for presence, and is of reverse order in alleleA/B.
    """
    AA,AB,BB = x.iloc[0],x.iloc[1],x.iloc[2]

    if newAT:
        dose = (0*AA)+ (1*AB) + (2*BB) # because in new AT, allele B is T for present.
    else:
        dose = (2*AA)+ (1*AB) + (0*BB) # makes allele A the effect allele and allle B the reference.
    return dose
def makedosage(dataframe):
    """
    Takes just dataframe of markers (as index) and sample IDs (as columns). FOR AA
    """
    samplenames = dataframe.columns[2:]
    absamp = dataframe[["alleleA", "alleleB"]]
    ATchecks = absamp.apply(checkAT, axis=1)

    df = dataframe[samplenames].T.copy()
    df["samples"] = df.index
    df.samples = df.samples.apply(lambda x : x.split('.')[0])

    dosedf = {}
    for samp in df.samples.unique():
        sampdf = df[df.samples==samp]
        sampdf = sampdf.append(ATchecks, ignore_index=True)
        dose = sampdf.apply(dosage, axis=0)
        dosedf[samp]=dose

    dosedf = pd.DataFrame(dosedf)
    dosedf = dosedf.drop("samples", axis=0)

    return dosedf

def makedosage_(dataframe):
    """
    Takes just dataframe of markers (as index) and sample IDs (as columns). FOR SNP/HLA
    """
    confirmAT = checkAT_(dataframe["alleleA"],dataframe["alleleB"])

    samplenames = dataframe.columns[2:]

    df = dataframe[samplenames].T.copy()
    df["samples"] = df.index
    df.samples = df.samples.apply(lambda x : x.split('.')[0])

    dosedf = {}
    for samp in df.samples.unique():
        sampdf = df[df.samples==samp]
        dose = dosage_(sampdf,confirmAT)
        dosedf[samp]=dose

    dosedf = pd.DataFrame(dosedf)
    dosedf = dosedf.drop("samples", axis=0)

    return dosedf

def checkAT_(colA, colB):
    "Basic check to see if it is using Michigan HLA imputation A/T absence/presence format. FOR SNP/HLA"
    if set(colA.unique()) == {"A"} and set(colB.unique()) == {"T"}:
        return True
    else:
        return False

def checkAT(df):
    "Basic check to see if it is using Michigan HLA imputation A/T absence/presence format. FOR AA"
    name_check=df.name
    if df["alleleA"]=="A" and df["alleleB"]=="T" and name_check.startswith("AA_"):
        return True
    else:
        return False

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

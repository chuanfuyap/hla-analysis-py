"""
Function to load and modify HLA dataset.

Currently support:
- [Beagle file](http://faculty.washington.edu/browning/beagle/b3.html) to process AA_ variant IDs.
- [SNP2HLA](https://software.broadinstitute.org/mpg/snp2hla/snp2hla_manual.html) dosage files.

If you have VCF file, please use Beagle's util files for converting to bgl/gprob files.
"""
class HLAdata:
    """
    The object type to store the all genotype information for HLA.
    """
    def __init__(self, ):
        self.HLA4digit = None
        self.SNPs = None
        self.AAs = None
        self.type = None # hardcall or probability/dosage

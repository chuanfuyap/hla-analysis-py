"""
init for `data` subpackage for hapy package.
"""

from hapy.data.io import *
from hapy.data.HLAdat import HLAdata

__all__ = ["read_famfile", "read_bgl", "read_gprobs", "read_dosage", "read_plinkraw", "HLAdat"]

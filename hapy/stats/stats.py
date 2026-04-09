"""
Backward-compatibility shim for older imports.

Prefer importing from `hapy.stats`:
  - analyse_aa / analyse_hla / analyse_snp / interaction
"""
from .api import analyse_aa, analyse_hla, analyse_snp, interaction
from .types import StandardConfig, InteractionConfig

__all__ = [
    "analyse_aa", "analyse_hla", "analyse_snp", "interaction",
    "StandardConfig", "InteractionConfig",
]
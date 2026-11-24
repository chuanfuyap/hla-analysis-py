"""
hapy.stats

Unified analysis entrypoints:
- analyse(kind=...)      # linear/logit
- interaction(...)       # linear/logit
- survival(kind=...)     # Cox PH

This module exports configs and filter utilities.
"""

from .api import analyse, analyse_many, interaction, survival
from .stepwise import stepwise
from .compare import compare_positions, compare_positions_motif
from .motif import analyse_motif, expand_motif
from .ld import LD
from .types import StandardConfig, InteractionConfig, SurvivalConfig
from . import filters as filters

__all__ = [
    "analyse",
    "analyse_motif",
    "expand_motif",
    "stepwise",
    "compare_positions",
    "compare_positions_motif",
    "LD",
    "analyse_many",
    "interaction",
    "survival",
    "StandardConfig",
    "InteractionConfig",
    "SurvivalConfig",
    "filters",
]
"""
hapy.stats

Unified analysis entrypoints:
- analyse(kind=...)      # linear/logit
- interaction(...)       # linear/logit
- survival(kind=...)     # Cox PH

This module exports configs and filter utilities.
"""

from .api import analyse, interaction, survival
from .types import StandardConfig, InteractionConfig, SurvivalConfig
from . import filters as filters

__all__ = [
    "analyse",
    "interaction",
    "survival",
    "StandardConfig",
    "InteractionConfig",
    "SurvivalConfig",
    "filters",
]
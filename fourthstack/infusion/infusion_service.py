"""Infusion service coordinating merge strategies."""
from .merge_strategies import simple_merge

def infuse(real, synthetic, strategy=None):
    """Infuse synthetic into real using the chosen strategy (stub)."""
    return simple_merge(real, synthetic)

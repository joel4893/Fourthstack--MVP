"""High-level generator pipeline combining models."""

from .gan_model import GANModel
from .diffusion_model import DiffusionModel


def generate_synthetic(n, config=None):
    """Run a simple pipeline to generate synthetic data (stub)."""
    gan = GANModel(config)
    dm = DiffusionModel(config)
    # naive: combine outputs
    return gan.generate(n) or dm.sample()

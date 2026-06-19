"""Composable physics components for :class:`ExoJaxPhysicalModel`.

Each component encapsulates one piece of atmospheric physics
(temperature-pressure profile, line absorption, Rayleigh scattering,
clouds, surface) and exposes a uniform contract:

  - ``compute_*(per_planet_args, ...) -> Contribution``

where :class:`Contribution` carries the per-layer optical-property
arrays (``dtau_total``, ``dtau_scatter``, ``g_weighted_num``) that the
atmosphere combines before passing to the 2-stream RT solver. Surface
and T-P components have analogous compute methods that return their
respective quantities.

To swap physics: replace one component instance with another that
implements the same contract (e.g. ``GrayCloud`` -> ``NoCloud``,
``RayleighScattering`` -> ``NullScattering``). Per-planet PyTree leaves
live on the components themselves, so HMC over their parameters
"just works" with ``eqx.filter_vmap`` or ``jax.vmap``.
"""

from .absorption import Absorption, PrecomputedAbsorption
from .base import (
    AbstractAbsorption,
    AbstractClouds,
    AbstractScattering,
    AbstractSurface,
    AbstractTPProfile,
    Contribution,
)
from .clouds import GrayCloud, NoCloud
from .mie_cloud import MieCloud, build_mie_cloud
from .mmr_profile import (
    AbstractMmrProfile,
    ConstantMmr,
    StratosphericPeakMmr,
    TroposphericMmr,
)
from .scattering import NullScattering, RayleighScattering
from .species import (
    BULK_GAS_RECIPES,
    MOLECULE_RECIPES,
    BulkGasRecipe,
    BulkGasResidual,
    MolecularSpecies,
    MoleculeRecipe,
    build_bulk_prebuilt,
    build_species_prebuilt,
)
from .surface import FlatSurface, WavelengthDependentSurface
from .tp import PowerLawTPProfile

__all__ = [
    "BULK_GAS_RECIPES",
    "MOLECULE_RECIPES",
    "Absorption",
    "AbstractAbsorption",
    "AbstractClouds",
    "AbstractMmrProfile",
    "AbstractScattering",
    "AbstractSurface",
    "AbstractTPProfile",
    "BulkGasRecipe",
    "BulkGasResidual",
    "ConstantMmr",
    "Contribution",
    "FlatSurface",
    "GrayCloud",
    "MieCloud",
    "MolecularSpecies",
    "MoleculeRecipe",
    "NoCloud",
    "NullScattering",
    "PowerLawTPProfile",
    "PrecomputedAbsorption",
    "RayleighScattering",
    "StratosphericPeakMmr",
    "TroposphericMmr",
    "WavelengthDependentSurface",
    "build_bulk_prebuilt",
    "build_mie_cloud",
    "build_species_prebuilt",
]

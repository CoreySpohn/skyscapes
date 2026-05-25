"""ExoJAX-backed physical-model components.

Self-contained subpackage holding everything specific to the ExoJAX
radiative-transfer backend: the :class:`ExoJaxPhysicalModel` orchestrator,
swappable physics components (T-P, absorption, scattering, clouds,
surface), PSG cross-section helpers, and Earth-epoch archetypes.

The backend-agnostic :class:`AbstractPhysicalModel` interface and the
simple non-RT physical models (Lambertian, grid) live one level up in
:mod:`skyscapes.physical_model`.
"""

from __future__ import annotations

from .components import (
    MOLECULE_RECIPES,
    Absorption,
    AbstractAbsorption,
    AbstractClouds,
    AbstractMmrProfile,
    AbstractScattering,
    AbstractSurface,
    AbstractTPProfile,
    BulkGasRecipe,
    BulkGasResidual,
    ConstantMmr,
    Contribution,
    FlatSurface,
    GrayCloud,
    MieCloud,
    MolecularSpecies,
    MoleculeRecipe,
    NoCloud,
    NullScattering,
    PowerLawTPProfile,
    RayleighScattering,
    StratosphericPeakMmr,
    TroposphericMmr,
    WavelengthDependentSurface,
    build_bulk_prebuilt,
    build_mie_cloud,
    build_species_prebuilt,
)
from .o3_chappuis import O3_MOLMASS, O3ChappuisOpacity
from .physical_model import (
    DEFAULT_MOLECULES,
    ExoJaxPhysicalModel,
    build_exojax_engines,
)
from .presets import (
    EARTH_ARCHEAN_VMRS,
    EARTH_EARLY_PROTEROZOIC_VMRS,
    EARTH_LATE_PROTEROZOIC_VMRS,
    EARTH_MODERN_VMRS,
    vmr_dict_to_earth_profile_dict,
    vmr_dict_to_log_mmr_dict,
    vmr_dict_to_mmr_dict,
)
from .psg_xs import PsgCrossSectionOpacity

__all__ = [
    "DEFAULT_MOLECULES",
    "EARTH_ARCHEAN_VMRS",
    "EARTH_EARLY_PROTEROZOIC_VMRS",
    "EARTH_LATE_PROTEROZOIC_VMRS",
    "EARTH_MODERN_VMRS",
    "MOLECULE_RECIPES",
    "O3_MOLMASS",
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
    "ExoJaxPhysicalModel",
    "FlatSurface",
    "GrayCloud",
    "MieCloud",
    "MolecularSpecies",
    "MoleculeRecipe",
    "NoCloud",
    "NullScattering",
    "O3ChappuisOpacity",
    "PowerLawTPProfile",
    "PsgCrossSectionOpacity",
    "RayleighScattering",
    "StratosphericPeakMmr",
    "TroposphericMmr",
    "WavelengthDependentSurface",
    "build_bulk_prebuilt",
    "build_exojax_engines",
    "build_mie_cloud",
    "build_species_prebuilt",
    "vmr_dict_to_earth_profile_dict",
    "vmr_dict_to_log_mmr_dict",
    "vmr_dict_to_mmr_dict",
]

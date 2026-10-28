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
    EmittingSurface,
    FlatSurface,
    GrayCloud,
    MieCloud,
    MolecularSpecies,
    MoleculeRecipe,
    NoCloud,
    NullScattering,
    PolynomialTPProfile,
    PowerLawTPProfile,
    RayleighScattering,
    StratosphericPeakMmr,
    TroposphericMmr,
    WavelengthDependentSurface,
    build_bulk_prebuilt,
    build_mie_cloud,
    build_species_prebuilt,
)
from .emission import (
    DEFAULT_EMISSION_MOLECULES,
    ExoJaxEmissionModel,
    brightness_temperature,
    build_exojax_emission_engines,
    log_gravity_cgs_from_mass_radius,
)
from .o3_chappuis import O3_MOLMASS, O3ChappuisOpacity
from .physical_model import (
    DEFAULT_MOLECULES,
    ExoJaxPhysicalModel,
    build_exojax_engines,
    grid_resolving_power,
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
    "DEFAULT_EMISSION_MOLECULES",
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
    "EmittingSurface",
    "ExoJaxEmissionModel",
    "ExoJaxPhysicalModel",
    "FlatSurface",
    "GrayCloud",
    "MieCloud",
    "MolecularSpecies",
    "MoleculeRecipe",
    "NoCloud",
    "NullScattering",
    "O3ChappuisOpacity",
    "PolynomialTPProfile",
    "PowerLawTPProfile",
    "PsgCrossSectionOpacity",
    "RayleighScattering",
    "StratosphericPeakMmr",
    "TroposphericMmr",
    "WavelengthDependentSurface",
    "brightness_temperature",
    "build_bulk_prebuilt",
    "build_exojax_emission_engines",
    "build_exojax_engines",
    "build_mie_cloud",
    "build_species_prebuilt",
    "grid_resolving_power",
    "log_gravity_cgs_from_mass_radius",
    "vmr_dict_to_earth_profile_dict",
    "vmr_dict_to_log_mmr_dict",
    "vmr_dict_to_mmr_dict",
]

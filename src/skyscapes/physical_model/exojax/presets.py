"""Atmospheric-composition archetypes (VMR / MMR conversion helpers).

Each preset is a dict mapping molecule name to volume mixing ratio
(``VMR`` = mole fraction, the convention used in atmospheric science
literature). :func:`vmr_dict_to_log_mmr_dict` converts to log10
mass-mixing ratios at construction time using a bulk gas (default
N2) to fill the residual, returning the dict shape
``ExoJaxPhysicalModel.from_default_setup`` consumes for ``log_mmrs``.

The four Earth-epoch presets bracket the modern oxygenated atmosphere
plus three earlier states (Archean, Early Proterozoic, Late
Proterozoic) drawn from a published "Earth through time" biosignature
table. Source attribution lives with whoever updates these values --
update the table here and note the source in your commit message or
local documentation.

I got these from Natasha Latouf and she has citations... somewhere.

Example::

    import jax.numpy as jnp
    from skyscapes.physical_model import ExoJaxPhysicalModel
    from skyscapes.physical_model.exojax.presets import (
        EARTH_MODERN_VMRS, vmr_dict_to_log_mmr_dict,
    )

    log_mmrs = vmr_dict_to_log_mmr_dict(EARTH_MODERN_VMRS, K=1)
    model = ExoJaxPhysicalModel.from_default_setup(
        log_mmrs=log_mmrs,
        T_eq_K=jnp.full((1,), 288.0),
        T_alpha=jnp.full((1,), 0.07),
        log_surface_albedo=jnp.full((1,), jnp.log10(0.3)),
        log_gravity_cgs=jnp.full((1,), jnp.log10(981.0)),
    )
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .components import (
    AbstractMmrProfile,
    ConstantMmr,
    StratosphericPeakMmr,
    TroposphericMmr,
)
from .components.species import (
    BULK_GAS_RECIPES,
)

# Approximate molar masses [g/mol]. For Rayleigh + opacity we use the
# value ExoJAX returns from its HITRAN-isotope-1 lookup or the
# Chappuis-O3 constant; the values below are used only for the
# VMR <-> MMR conversion in this module.
_MOLMASS_TABLE: dict[str, float] = {
    "H2O": 18.015,
    "CO2": 43.989,
    "CH4": 16.043,
    "O2": 31.989,
    "O3": 47.984,
    "CO": 27.995,
    "N2O": 44.013,
    "SO2": 63.962,
}


# ---------------------------------------------------------------------------
# Earth-epoch VMRs from an Earth-through-time table
# ---------------------------------------------------------------------------

EARTH_ARCHEAN_VMRS: dict[str, float] = {
    "H2O": 3.0e-3,
    "O2": 1.0e-8,
    "O3": 2.46e-10,
    "CH4": 7.0e-3,
    "CO2": 1.0e-1,
    "CO": 1.0e-4,
    "N2O": 1.0e-10,
    "SO2": 1.0e-7,
}

EARTH_EARLY_PROTEROZOIC_VMRS: dict[str, float] = {
    "H2O": 3.0e-3,
    "O2": 2.1e-4,
    "O3": 1.0e-8,
    "CH4": 1.65e-3,
    "CO2": 1.0e-2,
    "CO": 1.0e-5,
    "N2O": 8.37e-9,
    "SO2": 1.8e-8,
}

EARTH_LATE_PROTEROZOIC_VMRS: dict[str, float] = {
    "H2O": 3.0e-3,
    "O2": 2.1e-3,
    "O3": 7.0e-8,
    "CH4": 1.65e-3,
    "CO2": 1.0e-2,
    "CO": 1.0e-5,
    "N2O": 8.37e-9,
    "SO2": 1.8e-8,
}

EARTH_MODERN_VMRS: dict[str, float] = {
    "H2O": 3.0e-3,
    "O2": 2.1e-1,
    "O3": 7.0e-7,
    "CH4": 1.65e-6,
    "CO2": 3.8e-4,
    "CO": 1.0e-7,
    "N2O": 2.94e-7,
    "SO2": 1.0e-10,
}


# ---------------------------------------------------------------------------
# VMR <-> MMR conversion
# ---------------------------------------------------------------------------


def _mol_mass(name: str) -> float:
    if name in _MOLMASS_TABLE:
        return _MOLMASS_TABLE[name]
    raise KeyError(
        f"No molmass for {name!r}. Add it to "
        f"skyscapes.physical_model.exojax.presets._MOLMASS_TABLE."
    )


def vmr_dict_to_mmr_dict(
    vmrs: dict[str, float],
    bulk_gas: str = "N2",
) -> dict[str, float]:
    """Convert volume-mixing-ratio dict to mass-mixing-ratio dict.

    Computes mean molecular weight from the supplied VMRs plus the
    bulk-gas residual (mmr_bulk = 1 - sum(vmrs_tracked), in VMR
    space; then convert all to mass). Used internally by
    :func:`vmr_dict_to_log_mmr_dict`; exposed here for inspection.

    Args:
        vmrs: ``{name: VMR}`` for tracked molecules. Sum must be < 1
            (the residual is the bulk gas).
        bulk_gas: Name of the implicit residual gas (``"N2"`` for
            terrestrial atmospheres, ``"H2"`` for gas giants, ...).
            Must appear in :data:`BULK_GAS_RECIPES`.

    Returns:
        ``{name: MMR}`` for the tracked molecules. The bulk gas's
        MMR is *not* returned -- ``ExoJaxPhysicalModel`` computes it
        dynamically as the residual.
    """
    if bulk_gas not in BULK_GAS_RECIPES:
        raise KeyError(
            f"Unknown bulk gas {bulk_gas!r}. Known: {sorted(BULK_GAS_RECIPES.keys())}"
        )
    bulk_molmass = BULK_GAS_RECIPES[bulk_gas].molmass
    vmr_bulk = 1.0 - sum(vmrs.values())
    if vmr_bulk < 0:
        raise ValueError(
            f"sum(VMRs) = {sum(vmrs.values()):.3f} > 1; the bulk-gas "
            f"residual would be negative. Reduce tracked VMRs."
        )

    m_mean = (
        sum(vmrs[name] * _mol_mass(name) for name in vmrs) + vmr_bulk * bulk_molmass
    )

    return {name: vmr * _mol_mass(name) / m_mean for name, vmr in vmrs.items()}


def vmr_dict_to_log_mmr_dict(
    vmrs: dict[str, float],
    K: int = 1,
    bulk_gas: str = "N2",
) -> dict[str, Array]:
    """Convert VMR dict to log10 MMR dict with K-shaped arrays.

    Each value in the returned dict is shape ``(K,)`` so it can be
    passed directly as ``log_mmrs=`` to
    :meth:`ExoJaxPhysicalModel.from_default_setup`. The same VMR is used
    for every planet -- to give per-planet variation, modify the
    returned arrays before constructing the atmosphere.

    Every species gets a :class:`ConstantMmr` profile by default
    (uniform mmr at every altitude). For Earth-realistic profiles
    (O3 stratospheric peak, H2O cold-trap drop) use
    :func:`vmr_dict_to_earth_profile_dict` instead.
    """
    mmrs = vmr_dict_to_mmr_dict(vmrs, bulk_gas=bulk_gas)
    return {name: jnp.full((K,), float(jnp.log10(mmr))) for name, mmr in mmrs.items()}


# ---------------------------------------------------------------------------
# Earth-realistic mmr profile factory
# ---------------------------------------------------------------------------

# Tropopause pressure [bar] where Earth's cold trap drops H2O mmr by
# roughly three orders of magnitude. ~10 km altitude on Earth.
_H2O_COLD_TRAP_PRESSURE_BAR = 0.1
_H2O_COLD_TRAP_DECADES_DROP = 3.0  # mmr above is 1e-3 * mmr below
_H2O_COLD_TRAP_WIDTH_DECADES = 0.2  # sharpness of the sigmoid

# Earth's O3 column has a Gaussian-shaped concentration profile peaked
# in the stratosphere. Peak pressure ~10 mbar ~25--30 km altitude;
# half-decade Gaussian width.
_O3_PEAK_PRESSURE_BAR = 0.01
_O3_PEAK_SIGMA_DECADES = 0.5


def vmr_dict_to_earth_profile_dict(
    vmrs: dict[str, float],
    K: int = 1,
    bulk_gas: str = "N2",
) -> dict[str, AbstractMmrProfile]:
    """VMRs -> dict of Earth-realistic profiles per species.

    Most species get a :class:`ConstantMmr` (well-mixed assumption).
    For H2O and O3, where Earth's altitude profile differs sharply
    from uniform, the function builds:

      - **H2O**: :class:`TroposphericMmr` with the supplied VMR below
        the cold trap and a 3-decade drop above (parametrized at
        ``P = 0.1 bar``).
      - **O3**: :class:`StratosphericPeakMmr` with the supplied VMR as
        the peak value at ``P = 10 mbar`` (Gaussian width 0.5 dex).

    All values are taken from canonical Earth-atmosphere structure --
    customize by constructing your own profile dict if you need to
    fit per-planet profile parameters in an HMC retrieval.

    Args:
        vmrs: ``{name: VMR}``; H2O VMR is interpreted as the
            tropospheric value, O3 VMR as the stratospheric peak.
        K: Number of planets (per-planet broadcasting).
        bulk_gas: Implicit residual gas name.

    Returns:
        ``{name: AbstractMmrProfile}`` ready to pass as
        ``log_mmrs=`` to :meth:`ExoJaxPhysicalModel.from_default_setup`.
    """
    mmrs = vmr_dict_to_mmr_dict(vmrs, bulk_gas=bulk_gas)
    profiles: dict[str, AbstractMmrProfile] = {}
    for name, mmr in mmrs.items():
        log_mmr_val = float(jnp.log10(mmr))
        if name == "H2O":
            profiles[name] = TroposphericMmr(
                log_mmr_below=jnp.full((K,), log_mmr_val),
                log_mmr_above=jnp.full((K,), log_mmr_val - _H2O_COLD_TRAP_DECADES_DROP),
                log_pressure_step_bar=jnp.full(
                    (K,), float(jnp.log10(_H2O_COLD_TRAP_PRESSURE_BAR))
                ),
                log_transition_width_decades=jnp.full(
                    (K,), _H2O_COLD_TRAP_WIDTH_DECADES
                ),
            )
        elif name == "O3":
            profiles[name] = StratosphericPeakMmr(
                log_peak_mmr=jnp.full((K,), log_mmr_val),
                log_peak_pressure_bar=jnp.full(
                    (K,), float(jnp.log10(_O3_PEAK_PRESSURE_BAR))
                ),
                log_sigma_decades=jnp.full((K,), _O3_PEAK_SIGMA_DECADES),
            )
        else:
            profiles[name] = ConstantMmr(log_mmr=jnp.full((K,), log_mmr_val))
    return profiles


__all__ = [
    "EARTH_ARCHEAN_VMRS",
    "EARTH_EARLY_PROTEROZOIC_VMRS",
    "EARTH_LATE_PROTEROZOIC_VMRS",
    "EARTH_MODERN_VMRS",
    "vmr_dict_to_earth_profile_dict",
    "vmr_dict_to_log_mmr_dict",
    "vmr_dict_to_mmr_dict",
]

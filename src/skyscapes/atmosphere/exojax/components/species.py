"""Atmospheric-gas components.

Each :class:`MolecularSpecies` is a self-contained record for one
atmospheric molecule: its per-planet log mass-mixing ratio (a PyTree
leaf, fittable) plus the shared static data needed to evaluate both
line absorption and Rayleigh scattering for that species. The
atmosphere holds a tuple of these plus an optional
:class:`BulkGasResidual` for the implicit residual gas (N2 for
terrestrial atmospheres, H2 for gas giants).

To mix and match molecules: just change the tuple. To add a new
molecule supported by HITRAN, register its recipe in
:data:`MOLECULE_RECIPES`. The 8-molecule set used by Star+ "Earth
through time" (H2O, O2, O3, CH4, CO2, CO, N2O, SO2) is preregistered.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
from exojax.database import molinfo
from exojax.database.hitran.api import MdbHitran
from exojax.database.multimol import database_path_hitran12
from exojax.opacity import OpaPremodit
from exojax.opacity.opacont import OpaRayleigh
from jaxtyping import Array

from ..psg_xs import PsgCrossSectionOpacity
from .mmr_profile import AbstractMmrProfile

# ---------------------------------------------------------------------------
# PyTree records
# ---------------------------------------------------------------------------


class MolecularSpecies(eqx.Module):
    """One atmospheric molecule with an altitude-resolved mixing ratio.

    The mixing ratio is encoded as a profile component (e.g.
    :class:`~skyscapes.atmosphere.exojax.components.mmr_profile.ConstantMmr`
    for well-mixed gases, ``StratosphericPeakMmr`` for O3, ``TroposphericMmr``
    for H2O) rather than a single scalar, so altitude variation is
    represented explicitly.

    Attributes:
        profile: Per-species mmr profile. Owns the fittable
            log-mixing-ratio leaves.
        name: Molecule name, e.g. ``"H2O"``.
        molmass: Molar mass [g/mol].
        opa: Opacity engine. Either an ExoJAX ``OpaPremodit`` (for
            HITRAN-backed line-list molecules) or
            :class:`~skyscapes.atmosphere.exojax.o3_chappuis.O3ChappuisOpacity`
            (for visible cross-section absorbers like O3). May be
            ``None`` if the species contributes only Rayleigh scattering.
        rayleigh_xs: Pre-computed Rayleigh cross-section
            [cm^2/molecule] on the atmosphere's wavenumber grid, shape
            ``(n_nu,)``. Set to all-zeros to disable per-species
            Rayleigh contribution.
    """

    profile: AbstractMmrProfile
    name: str = eqx.field(static=True)
    molmass: float = eqx.field(static=True)
    opa: Any = eqx.field(static=True)
    rayleigh_xs: Array


class BulkGasResidual(eqx.Module):
    """Implicit residual gas filling the unallocated mass fraction.

    The mass-mixing ratio is computed dynamically as
    ``max(0, 1 - sum(tracked species mmrs))``. Contributes only to
    Rayleigh scattering (no line-list absorption is associated with
    the bulk gas in this model -- N2 is essentially transparent across
    300--1100 nm, H2/He likewise).

    Attributes:
        name: Gas name, e.g. ``"N2"`` for Earth or ``"H2"`` for gas giants.
        molmass: Molar mass [g/mol].
        rayleigh_xs: Rayleigh cross-section [cm^2/molecule] on the
            atmosphere's wavenumber grid, shape ``(n_nu,)``.
    """

    name: str = eqx.field(static=True)
    molmass: float = eqx.field(static=True)
    rayleigh_xs: Array


# ---------------------------------------------------------------------------
# Recipe registry (construction-time)
# ---------------------------------------------------------------------------

# N2O polarizability used for Rayleigh scattering. ExoJAX's
# polarizability table is missing N2O; the value here is the standard
# atmospheric-science value used in widely-cited Rayleigh references.
# Override at construction if a different source is preferred.
N2O_POLARIZABILITY_CM3 = 3.03e-24


@dataclass(frozen=True)
class MoleculeRecipe:
    """Build instructions for one molecule.

    Attributes:
        name: Molecule name (must match ExoJAX's HITRAN / polarizability
            keys when those tables apply).
        psg_xs_url: If set, use a PSG cross-section file at this URL
            (via :class:`~skyscapes.atmosphere.psg_xs.PsgCrossSectionOpacity`)
            instead of an ExoJAX line-list opa. Required for species
            whose absorption is dominated by electronic transitions in
            the visible/NIR (e.g. O3 Chappuis, SO2 UV).
        psg_xs_molmass: Molar mass [g/mol]; required when ``psg_xs_url``
            is set since we can't extract it from HITRAN in that case.
        polarizability_override: Explicit polarizability [cm^3] for the
            Rayleigh cross-section, used when ExoJAX's table doesn't
            have an entry. ``None`` lets ``OpaRayleigh`` look it up.
    """

    name: str
    psg_xs_url: str | None = None
    psg_xs_molmass: float | None = None
    polarizability_override: float | None = None


# PSG cross-section URLs for molecules where the visible/NIR absorption
# is dominated by electronic transitions (HITRAN line lists are sparse
# or absent in this range). See https://psg.gsfc.nasa.gov/helpatm.php
# for the full catalog.
PSG_O3_URL = "https://psg.gsfc.nasa.gov/data/linelists/xuv/data/o3.txt"
PSG_SO2_URL = "https://psg.gsfc.nasa.gov/data/linelists/xuv/data/so2.txt"

# The 8-molecule terrestrial-biosignature set. Order is not load-bearing:
# callers pick whichever subset they want via the ``molecules`` argument
# of ``build_exojax_engines``.
MOLECULE_RECIPES: dict[str, MoleculeRecipe] = {
    "H2O": MoleculeRecipe("H2O"),
    "CO2": MoleculeRecipe("CO2"),
    "CH4": MoleculeRecipe("CH4"),
    "O2": MoleculeRecipe("O2"),
    "O3": MoleculeRecipe("O3", psg_xs_url=PSG_O3_URL, psg_xs_molmass=47.99820),
    "CO": MoleculeRecipe("CO"),
    "N2O": MoleculeRecipe("N2O", polarizability_override=N2O_POLARIZABILITY_CM3),
    "SO2": MoleculeRecipe("SO2", psg_xs_url=PSG_SO2_URL, psg_xs_molmass=64.063),
}


@dataclass(frozen=True)
class BulkGasRecipe:
    """Build instructions for the implicit residual gas.

    Attributes:
        name: Gas name; must match an ExoJAX polarizability key (or
            provide ``polarizability_override``).
        molmass: Molar mass [g/mol].
        polarizability_override: Override polarizability if missing
            from ExoJAX's table.
    """

    name: str
    molmass: float
    polarizability_override: float | None = None


BULK_GAS_RECIPES: dict[str, BulkGasRecipe] = {
    "N2": BulkGasRecipe("N2", 28.0134),
    "H2": BulkGasRecipe("H2", 2.01588),
    "He": BulkGasRecipe("He", 4.002602),
}


def _build_rayleigh_xs(
    nu_grid: Array, name: str, polarizability_override: float | None
) -> Array:
    """Compute Rayleigh cross-section [cm^2/molecule] vs wavenumber."""
    ray = OpaRayleigh(nu_grid, molname=name)
    if polarizability_override is not None:
        ray.polarizability = polarizability_override
        ray.check_ready()
    return ray.xsvector()


def build_species_prebuilt(
    *,
    name: str,
    nu_grid: Array,
    nu_min: float,
    nu_max: float,
    databases_dir: str,
    crit: float,
) -> tuple[float, Any, Array]:
    """Construct (molmass, opa, rayleigh_xs) for one molecule.

    Used by ``build_exojax_engines``; not typically called by users
    directly. Returns the static parts of a :class:`MolecularSpecies`
    (everything except ``log_mmr``).
    """
    if name not in MOLECULE_RECIPES:
        raise KeyError(
            f"Unknown molecule {name!r}. Known: "
            f"{sorted(MOLECULE_RECIPES.keys())}. "
            f"Add a MoleculeRecipe entry to register new species."
        )
    recipe = MOLECULE_RECIPES[name]

    if recipe.psg_xs_url is not None:
        if recipe.psg_xs_molmass is None:
            raise ValueError(
                f"MoleculeRecipe for {name!r} has psg_xs_url but no "
                f"psg_xs_molmass; PSG xs path can't derive molmass from"
                f" the file alone."
            )
        opa = PsgCrossSectionOpacity(
            name=recipe.name,
            url=recipe.psg_xs_url,
            molmass=recipe.psg_xs_molmass,
            nu_grid=nu_grid,
            cache_dir=databases_dir,
        )
        molmass = recipe.psg_xs_molmass
    else:
        try:
            mdb = MdbHitran(
                str(Path(databases_dir) / database_path_hitran12(name)),
                nurange=[float(nu_min), float(nu_max)],
                isotope=1,
                crit=crit,
            )
        except ValueError as e:
            # HITRAN has no lines for this molecule in the requested
            # range (e.g. SO2 in the visible -- its strong bands are
            # UV/mid-IR; the visible overtones don't make HITRAN's
            # catalog). The species still contributes Rayleigh
            # scattering; absorption simply contributes zero.
            if "No line found" in str(e):
                warnings.warn(
                    f"{name!r} has no HITRAN lines in "
                    f"[{nu_min:.0f}, {nu_max:.0f}] cm^-1; "
                    f"using opa=None (zero absorption; Rayleigh "
                    f"contribution preserved).",
                    UserWarning,
                    stacklevel=2,
                )
                molmass = recipe_default_molmass(name)
                rayleigh_xs = _build_rayleigh_xs(
                    nu_grid,
                    recipe.name,
                    recipe.polarizability_override,
                )
                return molmass, None, rayleigh_xs
            raise
        opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, auto_trange=[100.0, 700.0])
        molmass = float(mdb.molmass)

    rayleigh_xs = _build_rayleigh_xs(
        nu_grid, recipe.name, recipe.polarizability_override
    )
    return molmass, opa, rayleigh_xs


def recipe_default_molmass(name: str) -> float:
    """Fallback molar mass [g/mol] when HITRAN provides no entry.

    Uses ExoJAX's ``molinfo.molmass`` lookup. Used when the species
    has no HITRAN lines in the configured wavenumber range and we
    therefore can't construct an MdbHitran to get the value from.
    """
    return float(molinfo.molmass(name))


def build_bulk_prebuilt(*, name: str, nu_grid: Array) -> tuple[float, Array]:
    """Construct (molmass, rayleigh_xs) for the implicit residual gas."""
    if name not in BULK_GAS_RECIPES:
        raise KeyError(
            f"Unknown bulk gas {name!r}. Known: "
            f"{sorted(BULK_GAS_RECIPES.keys())}. "
            f"Add a BulkGasRecipe entry to register a new bulk gas."
        )
    recipe = BULK_GAS_RECIPES[name]
    rayleigh_xs = _build_rayleigh_xs(
        nu_grid, recipe.name, recipe.polarizability_override
    )
    return recipe.molmass, rayleigh_xs

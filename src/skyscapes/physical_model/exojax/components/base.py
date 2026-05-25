"""Abstract base classes for atmosphere components.

Each abstract class declares the contract its concrete subclasses must
implement. Returning a structured :class:`Contribution` (rather than a
bare ``dtau``) lets the atmosphere combine absorption, Rayleigh, and
cloud contributions uniformly without each component caring whether the
others are present.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
from jaxtyping import Array

from .species import BulkGasResidual, MolecularSpecies


class Contribution(NamedTuple):
    """Per-layer optical-property contribution from one component.

    Fields:
        dtau_total: Layer optical depth this component adds to the total
            opacity budget, shape ``(n_layers, n_nu)``.
        dtau_scatter: Subset of ``dtau_total`` that is *scattering*
            (non-absorbing), shape ``(n_layers, n_nu)``. For a pure
            absorber this is zero. For Rayleigh ``ssa=1`` so it equals
            ``dtau_total``. For a partly-absorbing cloud with single-
            scattering albedo ``ssa_c``, this is ``ssa_c * dtau_cloud``.
        g_weighted_num: ``g * dtau_scatter`` numerator for the weighted-
            average asymmetry parameter, shape ``(n_layers, n_nu)``.
            Rayleigh contributes zero (isotropic, ``g=0``). A cloud with
            asymmetry ``g_c`` contributes ``g_c * ssa_c * dtau_cloud``.
    """

    dtau_total: Array
    dtau_scatter: Array
    g_weighted_num: Array


class AbstractTPProfile(eqx.Module, strict=True):
    """Temperature-pressure profile."""

    @abstractmethod
    def compute_Tarr(
        self, rt_engine, T_eq_K_scalar: Array, T_alpha_scalar: Array
    ) -> Array:
        """Return layer temperatures, shape ``(n_layers,)``."""


class AbstractAbsorption(eqx.Module, strict=True):
    """Per-layer absorption opacity from line lists / cross-sections.

    Concrete implementations iterate over the atmosphere's
    :class:`MolecularSpecies` tuple, calling each species' ``opa``
    engine and summing the contributions.
    """

    @abstractmethod
    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        Tarr: Array,
        pressure: Array,
        gravity: Array,
        rt_engine,
    ) -> Contribution:
        """Absorption contribution.

        ``dtau_scatter`` and ``g_weighted_num`` are zero for pure
        absorbers, but the return shape is the same as for scattering
        components so the atmosphere can combine them uniformly.
        """


class AbstractScattering(eqx.Module, strict=True):
    """Per-layer scattering opacity (e.g. Rayleigh)."""

    @abstractmethod
    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        bulk: BulkGasResidual | None,
        gravity: Array,
        rt_engine,
        n_layers: int,
        n_nu: int,
    ) -> Contribution:
        """Scattering contribution from tracked species + optional bulk gas.

        For a pure scatterer ``dtau_total == dtau_scatter``;
        ``g_weighted_num`` is zero for isotropic Rayleigh.
        """


class AbstractClouds(eqx.Module, strict=True):
    """Per-layer cloud opacity (mixed scattering + absorption)."""

    @abstractmethod
    def compute(
        self,
        log_pressure_bar_scalar: Array,
        log_opt_depth_scalar: Array,
        pressure: Array,
        n_nu: int,
    ) -> Contribution:
        """Cloud contribution.

        For a single-scattering-albedo cloud the scattering and
        absorption parts are both nonzero; for a purely scattering
        cloud ``dtau_total == dtau_scatter``.
        """


class AbstractSurface(eqx.Module, strict=True):
    """Bottom-of-atmosphere reflectivity."""

    @abstractmethod
    def compute_refl(self, log_albedo_scalar: Array, n_nu: int) -> Array:
        """Return surface reflectivity, shape ``(n_nu,)``."""

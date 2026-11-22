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

    Attributes:
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
        """Return layer temperatures, shape ``(n_layers,)``.

        Args:
            rt_engine: ExoJAX radiative-transfer engine; supplies the layer
                pressures.
            T_eq_K_scalar: Temperature parameter of one planet [K]. Concrete
                profiles define its role.
            T_alpha_scalar: Shape parameter of one planet. Concrete profiles
                define its role.

        Returns:
            Layer temperatures [K], shape ``(n_layers,)``.
        """


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

        Args:
            species: Species of one planet, each with its opacity engine and
                mass-mixing-ratio profile.
            Tarr: Layer temperatures [K], shape ``(n_layers,)``.
            pressure: Layer pressures [bar], shape ``(n_layers,)``.
            gravity: Surface gravity [cm/s^2], scalar.
            rt_engine: ExoJAX radiative-transfer engine; supplies the
                wavenumber grid and the opacity-to-column conversion.

        Returns:
            :class:`Contribution` of this component, fields of shape
            ``(n_layers, n_nu)``.
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

        Args:
            species: Species of one planet, each with its Rayleigh cross-section
                and mass-mixing-ratio profile.
            bulk: Implicit residual gas, or ``None`` for no bulk gas.
            gravity: Surface gravity [cm/s^2], scalar.
            rt_engine: ExoJAX radiative-transfer engine; supplies the layer
                pressures and the opacity-to-column conversion.
            n_layers: Number of atmospheric layers.
            n_nu: Number of wavenumber grid points.

        Returns:
            :class:`Contribution` of this component, fields of shape
            ``(n_layers, n_nu)``.
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

        Args:
            log_pressure_bar_scalar: Log10 cloud-deck pressure [bar] of one
                planet, scalar.
            log_opt_depth_scalar: Log10 vertically integrated cloud optical
                depth of one planet, scalar.
            pressure: Layer pressures [bar], shape ``(n_layers,)``.
            n_nu: Number of wavenumber grid points.

        Returns:
            :class:`Contribution` of this component, fields of shape
            ``(n_layers, n_nu)``.
        """


class AbstractSurface(eqx.Module, strict=True):
    """Bottom-of-atmosphere reflectivity."""

    @abstractmethod
    def compute_refl(self, log_albedo_scalar: Array, n_nu: int) -> Array:
        """Return surface reflectivity, shape ``(n_nu,)``.

        Args:
            log_albedo_scalar: Log10 surface albedo of one planet, scalar.
            n_nu: Number of wavenumber grid points.

        Returns:
            Surface reflectivity (dimensionless), shape ``(n_nu,)``.
        """

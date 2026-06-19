"""Absorption opacity components.

``Absorption`` iterates over the atmosphere's :class:`MolecularSpecies`
tuple and sums per-molecule line-list / cross-section contributions.
Each species owns its own opa engine and altitude-resolved mmr profile;
this component is a thin orchestrator with no per-molecule state of
its own.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractAbsorption, Contribution
from .species import MolecularSpecies


class Absorption(AbstractAbsorption):
    """Sum of per-species line-list / cross-section absorption.

    Iterates over the species tuple, skipping any with ``opa is None``
    (e.g. a species included purely for its Rayleigh contribution).
    """

    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        Tarr: Array,
        pressure: Array,
        gravity: Array,
        rt_engine,
    ) -> Contribution:
        """Sum per-species absorption optical depth."""
        dtau_per_mol = [
            rt_engine.opacity_profile_xs(
                s.opa.xsmatrix(Tarr, pressure),
                s.profile.evaluate(pressure),
                s.molmass,
                gravity,
            )
            for s in species
            if s.opa is not None
        ]
        if not dtau_per_mol:
            zeros = jnp.zeros((pressure.shape[0], rt_engine.nu_grid.shape[0]))
            return Contribution(
                dtau_total=zeros, dtau_scatter=zeros, g_weighted_num=zeros
            )
        dtau = jnp.stack(dtau_per_mol, axis=0).sum(axis=0)
        zeros = jnp.zeros_like(dtau)
        return Contribution(dtau_total=dtau, dtau_scatter=zeros, g_weighted_num=zeros)


class PrecomputedAbsorption(AbstractAbsorption):
    """Absorption with per-species ``xsmatrix`` precomputed at a fixed TP.

    Drop-in for :class:`Absorption` for retrievals: holds each absorbing species'
    cross-section matrix (computed once via ``opa.xsmatrix`` at the fixed
    temperature-pressure structure) and applies only the live mmr scaling. The
    spectrum is therefore differentiable in abundance but INERT IN TEMPERATURE --
    the stored ``xsmatrix`` bakes ``Tarr``, so ``tp_profile`` leaves no longer
    change the output. To fit temperature, use the full recompute path
    (``from_default_setup``) instead. Build via
    ``ExoJaxPhysicalModel.for_retrieval``.

    ``xsmatrix_per_species`` corresponds, in order, to the species with
    ``opa is not None`` (the same ones :class:`Absorption` would sum over).
    """

    xsmatrix_per_species: tuple[Array, ...]

    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        Tarr: Array,
        pressure: Array,
        gravity: Array,
        rt_engine,
    ) -> Contribution:
        """Sum per-species absorption from the stored cross-sections."""
        absorbing = [s for s in species if s.opa is not None]
        dtau_per_mol = [
            rt_engine.opacity_profile_xs(
                xs, s.profile.evaluate(pressure), s.molmass, gravity
            )
            for xs, s in zip(self.xsmatrix_per_species, absorbing, strict=True)
        ]
        if not dtau_per_mol:
            zeros = jnp.zeros((pressure.shape[0], rt_engine.nu_grid.shape[0]))
            return Contribution(
                dtau_total=zeros, dtau_scatter=zeros, g_weighted_num=zeros
            )
        dtau = jnp.stack(dtau_per_mol, axis=0).sum(axis=0)
        zeros = jnp.zeros_like(dtau)
        return Contribution(dtau_total=dtau, dtau_scatter=zeros, g_weighted_num=zeros)

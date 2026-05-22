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

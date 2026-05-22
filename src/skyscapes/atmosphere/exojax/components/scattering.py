"""Scattering opacity components."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractScattering, Contribution
from .species import BulkGasResidual, MolecularSpecies


class RayleighScattering(AbstractScattering):
    """Rayleigh scattering from tracked species + optional bulk residual.

    Iterates over the atmosphere's species tuple plus the bulk gas (if
    present), computing each gas's contribution from its own
    ``rayleigh_xs`` and ``molmass``. The bulk gas's mass-mixing ratio
    is computed dynamically as ``max(0, 1 - sum(tracked mmrs))``.

    To disable per-species Rayleigh while keeping the bulk: set the
    species' ``rayleigh_xs`` to zeros. To disable scattering entirely:
    use :class:`NullScattering`.
    """

    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        bulk: BulkGasResidual | None,
        gravity: Array,
        rt_engine,
        n_layers: int,
        n_nu: int,
    ) -> Contribution:
        """Sum tracked-species + bulk-gas Rayleigh contributions.

        Bulk gas mmr is computed per-layer as
        ``max(0, 1 - sum(species profiles at this pressure))``, so
        altitude variation of the tracked species translates into
        altitude variation of the bulk-gas residual.
        """
        pressure = rt_engine.pressure

        def _ray(sigma_nu, mmr_profile, molmass):
            return rt_engine.opacity_profile_xs(
                jnp.broadcast_to(sigma_nu[None, :], (n_layers, n_nu)),
                mmr_profile,
                molmass,
                gravity,
            )

        dtau_tracked = jnp.stack(
            [
                _ray(s.rayleigh_xs, s.profile.evaluate(pressure), s.molmass)
                for s in species
            ],
            axis=0,
        ).sum(axis=0)

        if bulk is not None:
            total_tracked = jnp.stack(
                [s.profile.evaluate(pressure) for s in species],
                axis=0,
            ).sum(axis=0)  # (n_layers,)
            mmr_bulk_profile = jnp.maximum(0.0, 1.0 - total_tracked)
            dtau_bulk = _ray(bulk.rayleigh_xs, mmr_bulk_profile, bulk.molmass)
            dtau = dtau_tracked + dtau_bulk
        else:
            dtau = dtau_tracked

        # Rayleigh: pure scattering (ssa=1), isotropic (g=0).
        zeros = jnp.zeros_like(dtau)
        return Contribution(dtau_total=dtau, dtau_scatter=dtau, g_weighted_num=zeros)


class NullScattering(AbstractScattering):
    """Disable scattering entirely: contributes zero opacity.

    Useful for ablation studies (e.g. quantifying how much Rayleigh
    affects a retrieval) and for atmospheres where scattering is
    negligible.
    """

    def compute(
        self,
        species: tuple[MolecularSpecies, ...],
        bulk: BulkGasResidual | None,
        gravity: Array,
        rt_engine,
        n_layers: int,
        n_nu: int,
    ) -> Contribution:
        """Return zero-everywhere contribution."""
        zeros = jnp.zeros((n_layers, n_nu))
        return Contribution(dtau_total=zeros, dtau_scatter=zeros, g_weighted_num=zeros)

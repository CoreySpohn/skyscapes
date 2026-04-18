"""scene.System -- the top-level PyTree container.

Holds a star, a heterogeneous tuple of planets, an optional disk, and the
Kepler-solver callable (static so JIT doesn't re-trace). The tuple makes
the system variadic: a 1-planet system and an 8-planet system have
compatible shapes as long as per-planet arrays broadcast correctly.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..disk import AbstractDisk
from .planet import Planet
from .star import AbstractStar


class System(eqx.Module):
    """Astrophysical scene: star + tuple of planets + optional disk.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        disk: Optional extended-source disk (``AbstractDisk | None``).
        trig_solver: Scalar Kepler-trig solver (static; see
            ``orbix.kepler.shortcuts.grid.get_grid_solver``). Required --
            callers must provide a built solver, not None.
    """

    star: AbstractStar
    planets: tuple[Planet, ...]
    trig_solver: Callable = eqx.field(static=True)
    disk: AbstractDisk | None = None

    @property
    def n_planets(self) -> int:
        """Total number of planets across all composed ``Planet`` modules."""
        return sum(p.n_planets for p in self.planets)

    def positions(self, t_jd: Array) -> Array:
        """Concatenated on-sky positions, shape ``(2, K_total, T)``."""
        per_planet = [
            p.position_arcsec(self.trig_solver, t_jd, star=self.star)
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=1)

    def contrasts(self, wavelength_nm: Array, t_jd: Array) -> Array:
        """Per-planet contrast, shape ``(K_total, T)``."""
        per_planet = [
            p.contrast(self.trig_solver, wavelength_nm, t_jd, star=self.star)
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=0)

    def planet_flux_densities(self, wavelength_nm: Array, t_jd: Array) -> Array:
        """Per-planet flux density [ph/s/m^2/nm], shape ``(K_total, T)``."""
        per_planet = [
            p.spec_flux_density(self.trig_solver, wavelength_nm, t_jd, star=self.star)
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=0)

    def alpha_dMag(self, t_jd: Array) -> tuple[Array, Array]:
        """Per-planet projected separation + dMag, each shape ``(K_total, T)``."""
        per_planet = [
            p.alpha_dMag(self.trig_solver, t_jd, star=self.star) for p in self.planets
        ]
        alpha = jnp.concatenate([a for a, _ in per_planet], axis=0)
        dMag = jnp.concatenate([m for _, m in per_planet], axis=0)
        return alpha, dMag

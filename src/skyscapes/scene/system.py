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

from .._repr import indent
from ..disk import AbstractDisk
from .planet import Planet
from .star import AbstractStar


class System(eqx.Module):
    """Astrophysical scene: star + tuple of planets + optional disk.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        trig_solver: Scalar Kepler-trig solver (static; see
            ``orbix.kepler.shortcuts.grid.get_grid_solver``). Required --
            callers must provide a built solver, not None.
        disk: Optional extended-source disk (``AbstractDisk | None``).
        midplane_inc_deg: System midplane inclination [deg] in the
            barycentric -> sky frame. Default 0.0 means "midplane = sky"
            and is intentionally indistinguishable from a real face-on
            system at this inclination; this ambiguity is acceptable
            because the field is diagnostic-only after load (no runtime
            hot path consults it). Populated by ``io.from_exovista``
            from the FITS star header. After load, frame rotation has
            already been baked into each ``Planet``'s orbital elements.
        midplane_pa_deg: System midplane position angle [deg]. Same
            semantics as ``midplane_inc_deg``.
    """

    star: AbstractStar
    planets: tuple[Planet, ...]
    trig_solver: Callable = eqx.field(static=True)
    disk: AbstractDisk | None = None
    midplane_inc_deg: float = 0.0
    midplane_pa_deg: float = 0.0

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

    def __repr__(self) -> str:
        """Tree-shaped summary: star + planets + disk + midplane geometry."""
        n_modules = len(self.planets)
        n_total = self.n_planets
        lines = [
            (
                f"System(n_planet_modules={n_modules}, "
                f"K_total={n_total}, "
                f"midplane: i={self.midplane_inc_deg:.2f} deg, "
                f"PA={self.midplane_pa_deg:.2f} deg)"
            ),
            indent("star: " + repr(self.star)),
        ]
        for i, p in enumerate(self.planets):
            lines.append(indent(f"planet[{i}]:"))
            lines.append(indent(repr(p), prefix="    "))
        if self.disk is not None:
            lines.append(indent("disk: " + repr(self.disk)))
        else:
            lines.append("  disk: None")
        return "\n".join(lines)

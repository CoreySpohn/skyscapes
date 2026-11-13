"""scene.System -- the top-level PyTree container.

Holds a star, a heterogeneous tuple of planets, an optional disk, and the
Kepler-solver callable (static so JIT doesn't re-trace). The tuple makes
the system variadic: a 1-planet system and an 8-planet system have
compatible shapes as long as per-planet arrays broadcast correctly.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._repr import indent
from ..disk import AbstractDisk
from ..spectral import _model_max_knots, bin_mean
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
        midplane_inc_deg: Disk midplane inclination [deg] in [0, 180],
            with 0 face-on (the default). Together with ``midplane_pa_deg``,
            orients parametric disks and visualization outlines in the
            right-handed (east, north, away) sky frame. ``io.from_exovista``
            preserves the FITS star header's I. Planet orbits and loaded
            disk images already carry their own geometry.
        midplane_pa_deg: Disk line-of-nodes angle [deg], measured from east
            (+x) toward north (+y), default 0. This is not astronomical PA.
            ``io.from_exovista`` converts the FITS star header's west-based
            PA to ``(-PA) % 360``, preserving the near half toward the observer (-z).
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

    def positions(self, time_jd: Array) -> Array:
        """Concatenated on-sky positions, shape ``(2, K_total, T)``."""
        per_planet = [
            p.position_arcsec(self.trig_solver, time_jd, star=self.star)
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=1)

    def contrasts(self, wavelength_nm: Array, time_jd: Array) -> Array:
        """Per-planet contrast at a scalar wavelength [nm], shape ``(K_total, T)``."""
        if jnp.size(wavelength_nm) != 1:
            raise ValueError("need one wavelength; use contrast_cube")
        per_planet = [
            p.contrast(self.trig_solver, wavelength_nm, time_jd, star=self.star)
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=0)

    def contrast_cube(self, wavelengths_nm: Array, time_jd: Array) -> Array:
        """Per-planet contrast at each wavelength, shape ``(W, K_total, T)``.

        Args:
            wavelengths_nm: Wavelengths [nm], shape ``(W,)``.
            time_jd: Observation Julian Dates, shape ``(T,)``.
        """
        return jax.vmap(lambda w: self.contrasts(w, time_jd))(
            jnp.asarray(wavelengths_nm)
        )

    def bin_mean_contrast_cube(
        self, centers_nm: Array, widths_nm: Array, time_jd: Array, *, n_nodes: int = 4
    ) -> Array:
        """Ratio of bin-integrated planet and star photons, ``(W, K_total, T)``.

        Centers and full nonnegative widths are ``(W,)`` in nm; dates are
        ``(T,)``. Zero width returns ``contrast_cube``. For linear stellar
        tables each stellar interval is integrated with ``n_nodes`` Gauss-
        Legendre nodes, clipping the end intervals to the bin. This resolves
        stellar lines and integrates the product of stellar flux and planet
        contrast, with an exact stellar denominator. Resolve any additional
        planetary features by increasing ``n_nodes``. Other stars use one
        Gauss-Legendre rule per bin. This is a flux-weighted contrast, not
        a mean of monochromatic contrasts. See Davis and Rabinowitz (1984),
        Methods of Numerical Integration, for the quadrature rule. Concrete
        bin geometry determines the static interval cap; traced geometry uses
        a conservative full-table cap. Close over concrete bins when compiling
        to retain bounded work.
        """
        numerator = self.bin_mean_planet_flux_densities(
            centers_nm, widths_nm, time_jd, n_nodes=n_nodes
        )
        knots = self.star.quadrature_knots_nm
        denominator = bin_mean(
            lambda w: jax.vmap(lambda t: self.star.spec_flux_density(w, t))(time_jd),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=_model_max_knots(centers_nm, widths_nm, knots),
        )
        centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
        point_bin = centers - widths / 2 == centers + widths / 2
        safe_denominator = jnp.where(point_bin[:, None], 1.0, denominator)
        ratio = numerator / safe_denominator[:, None, :]
        return jnp.where(
            point_bin[:, None, None], self.contrast_cube(centers, time_jd), ratio
        )

    def bin_mean_planet_flux_densities(
        self, centers_nm: Array, widths_nm: Array, time_jd: Array, *, n_nodes: int = 4
    ) -> Array:
        """Band-mean planet flux [ph/s/m^2/nm], shape ``(W, K_total, T)``.

        Centers and nonnegative full widths are ``(W,)`` in nm; dates are
        ``(T,)``. Integrate contrast times stellar flux, splitting at the star's
        public quadrature knots. Zero width returns the monochromatic flux.
        Use Gauss-Legendre quadrature (Davis and Rabinowitz 1984, Methods of
        Numerical Integration). Concrete geometry determines the static cap;
        traced geometry uses a conservative full-table cap. Close over concrete
        bins when compiling to retain bounded work.
        """
        knots = self.star.quadrature_knots_nm
        return bin_mean(
            lambda w: self.planet_flux_densities(w, time_jd),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=_model_max_knots(centers_nm, widths_nm, knots),
        )

    def planet_flux_densities(self, wavelength_nm: Array, time_jd: Array) -> Array:
        """Per-planet flux density [ph/s/m^2/nm], shape ``(K_total, T)``."""
        per_planet = [
            p.spec_flux_density(
                self.trig_solver, wavelength_nm, time_jd, star=self.star
            )
            for p in self.planets
        ]
        return jnp.concatenate(per_planet, axis=0)

    def alpha_dMag(
        self, time_jd: Array, *, wavelength_nm: float = 600.0
    ) -> tuple[Array, Array]:
        """Per-planet separation + dMag at wavelength_nm (default 600 nm).

        Each result has shape ``(K_total, T)``; wavelength_nm is scalar.
        """
        per_planet = [
            p.alpha_dMag(
                self.trig_solver, time_jd, star=self.star, wavelength_nm=wavelength_nm
            )
            for p in self.planets
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

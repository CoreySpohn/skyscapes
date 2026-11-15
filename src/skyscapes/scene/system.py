"""scene.System -- the top-level PyTree container.

Holds a star, a heterogeneous tuple of planets, an optional disk, and the
legacy Kepler-solver callable (static so JIT does not re-trace). The tuple makes
the system variadic: a 1-planet system and an 8-planet system have
compatible shapes as long as per-planet arrays broadcast correctly.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import Msun2kg
from jaxtyping import Array
from orbix.kepler import Markley
from orbix.system import Body, Ephemeris, SkyFrame
from orbix.system import System as DynamicsSystem

from .._repr import indent
from ..disk import AbstractDisk
from ..spectral import Bandpass, _model_max_knots, bin_mean
from .planet import Planet
from .star import AbstractStar


class _SceneEphemeris(Ephemeris):
    """Chain-order states with body indices for planets in scene order.

    ``scene_planet_indices`` has shape ``(K_total,)`` and indexes the body
    axis of all inherited observables; body zero is always the star.
    """

    scene_planet_indices: Array


class System(eqx.Module):
    """Astrophysical scene: star + tuple of planets + optional disk.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        trig_solver: Required legacy static solver field, unused by System methods.
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

    def _chain_order(self) -> Array:
        """Stable inner-to-outer permutation of the flattened scene planets."""
        sizes = (
            jnp.concatenate([p.orbit.a_AU for p in self.planets])
            if self.planets
            else jnp.empty(0)
        )
        return jnp.argsort(sizes, stable=True)

    def dynamics(self) -> DynamicsSystem:
        """Build a Jacobi chain from the current scene leaves, without host conversion.

        Each orbit's ``t0_d`` is the epoch at which its mean anomaly is
        ``M0_rad``: the existing two-body convention is
        ``M(t) = M0_rad + n * (time_jd - t0_d)``. Map ``t0_d`` directly
        to ``epoch_d``, retain ``M0_rad``, and pass Julian dates unchanged
        as ``time_d``; SkyFrame has a zero date origin. Jacobi rates use
        the enclosed mass, including the current planet. Masses are in
        solar masses; nonfinite or nonpositive planet masses become zero.
        Bodies are stably sorted by semi-major axis, inner to outer.
        """
        order = self._chain_order()
        elements = {
            name: jnp.concatenate([getattr(p.orbit, name) for p in self.planets])[order]
            if self.planets
            else jnp.empty(0)
            for name in ("a_AU", "e", "i_rad", "W_rad", "w_rad", "M0_rad", "t0_d")
        }
        masses = (
            jnp.concatenate([p.planet_mass_kg() / Msun2kg for p in self.planets])[order]
            if self.planets
            else jnp.empty(0)
        )
        return DynamicsSystem.from_classical(
            Body(self.star.Ms_kg / Msun2kg),
            Body(masses),
            SkyFrame(self.star.dist_pc),
            epoch_d=elements.pop("t0_d"),
            **elements,
        )

    def ephemeris(self, time_jd: Array, *, solver=None) -> _SceneEphemeris:
        """Solve once with Markley by default and retain scene-order body indices.

        The returned orbix Ephemeris stores ``(K_total+1, T, 3)`` states
        in chain order. Its ``scene_planet_indices`` array addresses the
        planets in scene order. Pass this object as ``ephemeris=`` to any
        geometry consumer to reuse the solve; dates must match that solve.
        ``solver`` accepts an orbix.kepler strategy, independent of the
        retained legacy ``trig_solver`` field.
        """
        solved = self.dynamics().solve(
            time_jd, solver=Markley() if solver is None else solver
        )
        return _SceneEphemeris(
            solved.position_AU,
            solved.velocity_AU_per_d,
            solved.dist_pc,
            jnp.argsort(self._chain_order(), stable=True) + 1,
        )

    def positions(self, time_jd: Array, *, ephemeris=None) -> Array:
        """Star-relative on-sky positions, shape ``(2, K_total, T)``."""
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(eph.scene_planet_indices, 0), axis=0)

    def position_angles(self, time_jd: Array, *, ephemeris=None) -> Array:
        """Position angles east of north in ``[0, 360)``, ``(K_total, T)``."""
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        return eph.position_angle_deg(eph.scene_planet_indices, 0)

    def star_reflex(self, time_jd: Array, *, ephemeris=None) -> Array:
        """Stellar (east, north) barycentric offsets [arcsec], ``(2, T)``.

        The Jacobi chain includes interior planets in each orbit's reference
        mass and centre; it is a Kepler hierarchy, without mutual perturbations.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(0, "barycenter"), axis=0)

    def star_radial_velocity(self, time_jd: Array, *, ephemeris=None) -> Array:
        """Stellar barycentric radial velocity [m/s], positive receding, ``(T,)``.

        Read the Jacobi ephemeris without adding a systemic velocity.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        return eph.radial_velocity_m_per_s(0, "barycenter")

    def _photometry(self, wavelength_nm, time_jd, ephemeris, *, flux=False):
        """Dispatch physical models on star-relative phase/distance, each (K, T)."""
        if jnp.size(wavelength_nm) != 1:
            raise ValueError("need one wavelength; use contrast_cube")
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        phase = eph.phase_angle_rad(eph.scene_planet_indices, 0)
        distance = eph.distance_AU(eph.scene_planet_indices, 0)
        values = []
        start = 0
        for planet in self.planets:
            stop = start + planet.n_planets
            geometry = dict(
                phase_angle_rad=phase[start:stop],
                dist_AU=distance[start:stop],
                Rp_Rearth=planet.Rp_Rearth,
                time_jd=time_jd,
                star=self.star,
            )
            model = planet.physical_model
            values.append(
                model.spec_flux_density(wavelength_nm, **geometry)
                if flux
                else model.contrast_at_time(wavelength_nm=wavelength_nm, **geometry)
            )
            start = stop
        return (
            jnp.concatenate(values, axis=0) if values else jnp.empty((0, time_jd.size))
        )

    def contrasts(
        self, wavelength_nm: Array, time_jd: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet contrast at a scalar wavelength [nm], shape ``(K_total, T)``."""
        return self._photometry(wavelength_nm, time_jd, ephemeris)

    def contrast_cube(
        self, wavelengths_nm: Array, time_jd: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet contrast at each wavelength, shape ``(W, K_total, T)``.

        Args:
            wavelengths_nm: Wavelengths [nm], shape ``(W,)``.
            time_jd: Observation Julian Dates, shape ``(T,)``.
            ephemeris: Optional scene ephemeris solved at these dates.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        return jax.vmap(lambda w: self.contrasts(w, time_jd, ephemeris=eph))(
            jnp.asarray(wavelengths_nm)
        )

    def bin_mean_contrast_cube(
        self,
        centers_nm: Array,
        widths_nm: Array,
        time_jd: Array,
        *,
        n_nodes: int = 4,
        ephemeris=None,
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
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        numerator = self.bin_mean_planet_flux_densities(
            centers_nm, widths_nm, time_jd, n_nodes=n_nodes, ephemeris=eph
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
            point_bin[:, None, None],
            self.contrast_cube(centers, time_jd, ephemeris=eph),
            ratio,
        )

    def band_contrasts(
        self, bandpass: Bandpass, time_jd: Array, *, n_nodes: int = 4, ephemeris=None
    ) -> Array:
        """Ratio of band-integrated planet and star photons, ``(K_total, T)``.

        Multiply photon flux densities by the photon-counting transmission.
        Split quadrature at both the bandpass table and stellar spectral
        knots. ``n_nodes`` is the Gauss-Legendre order per interval: linear
        transmission is exact at order one, its product with a linear
        spectrum at order two. Increase the order to resolve smooth spectral
        curvature or additional planetary features. The integrated stellar
        photon flux must be nonzero.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        centers = jnp.atleast_1d((bandpass.low_nm + bandpass.high_nm) / 2)
        widths = jnp.atleast_1d(bandpass.high_nm - bandpass.low_nm)
        knots = bandpass.wavelength_nm
        if self.star.quadrature_knots_nm is not None:
            knots = jnp.sort(jnp.concatenate((knots, self.star.quadrature_knots_nm)))
        quadrature = dict(
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=_model_max_knots(centers, widths, knots),
            weight_fn=bandpass,
        )
        numerator = bin_mean(
            lambda w: self.planet_flux_densities(w, time_jd, ephemeris=eph),
            centers,
            widths,
            **quadrature,
        )[0]
        denominator = bin_mean(
            lambda w: jax.vmap(lambda t: self.star.spec_flux_density(w, t))(time_jd),
            centers,
            widths,
            **quadrature,
        )[0]
        return numerator / denominator[None, :]

    def bin_mean_planet_flux_densities(
        self,
        centers_nm: Array,
        widths_nm: Array,
        time_jd: Array,
        *,
        n_nodes: int = 4,
        ephemeris=None,
    ) -> Array:
        """Band-mean planet flux [ph/s/m^2/nm], shape ``(W, K_total, T)``.

        Centers and nonnegative full widths are ``(W,)`` in nm; dates are
        ``(T,)``. Integrate each model's photon flux, splitting at the star's
        public quadrature knots. Zero width returns the monochromatic flux.
        Use Gauss-Legendre quadrature (Davis and Rabinowitz 1984, Methods of
        Numerical Integration). Concrete geometry determines the static cap;
        traced geometry uses a conservative full-table cap. Close over concrete
        bins when compiling to retain bounded work.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        knots = self.star.quadrature_knots_nm
        return bin_mean(
            lambda w: self.planet_flux_densities(w, time_jd, ephemeris=eph),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=_model_max_knots(centers_nm, widths_nm, knots),
        )

    def planet_flux_densities(
        self, wavelength_nm: Array, time_jd: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet flux density [ph/s/m^2/nm], shape ``(K_total, T)``."""
        return self._photometry(wavelength_nm, time_jd, ephemeris, flux=True)

    def alpha_dMag(
        self, time_jd: Array, *, wavelength_nm: float = 600.0, ephemeris=None
    ) -> tuple[Array, Array]:
        """Per-planet separation + dMag at wavelength_nm (default 600 nm).

        Each result has shape ``(K_total, T)``; wavelength_nm is scalar.
        """
        eph = self.ephemeris(time_jd) if ephemeris is None else ephemeris
        alpha = eph.separation_arcsec(eph.scene_planet_indices, 0)
        contrast = self.contrasts(jnp.asarray(wavelength_nm), time_jd, ephemeris=eph)
        dMag = -2.5 * jnp.log10(contrast + jnp.finfo(contrast.dtype).tiny)
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

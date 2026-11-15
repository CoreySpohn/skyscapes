"""Photometric scenes with one orbix system owning all dynamical state."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.epoch import Epoch
from jaxtyping import Array
from orbix.kepler import Markley
from orbix.system import Body, Ephemeris, SkyFrame
from orbix.system import System as DynamicsSystem

from .._repr import indent
from ..disk import AbstractDisk
from ..spectral import Bandpass, _model_max_knots, bin_mean
from .planet import Planet
from .star import AbstractStar

_DEFAULT_EPOCH = Epoch()


class _SceneEphemeris(Ephemeris):
    """Chain-order states with body indices for planets in scene order.

    ``scene_planet_indices`` has shape ``(K_total,)`` and indexes the body
    axis of all inherited observables; body zero is always the star.
    """

    scene_planet_indices: Array


def _known_planet_mass(mass_Msun):
    """Use zero mass for a nonfinite or nonpositive catalogue value."""
    mass = jnp.asarray(mass_Msun)
    return jnp.where(jnp.isfinite(mass) & (mass > 0), mass, 0.0)


class System(eqx.Module):
    """Astrophysical scene: star + tuple of planets + optional disk.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        dynamics: Orbix Jacobi chain, declared inner to outer.
        body_index: Static permutation mapping flattened scene planet indices to
            zero-based chain planet indices. Defaults to the identity.
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
    dynamics: DynamicsSystem
    disk: AbstractDisk | None = None
    midplane_inc_deg: float = 0.0
    midplane_pa_deg: float = 0.0
    body_index: tuple[int, ...] = eqx.field(static=True, default=())

    @property
    def n_planets(self) -> int:
        """Total number of planets across all composed ``Planet`` modules."""
        return sum(p.n_planets for p in self.planets)

    def __init__(
        self,
        star,
        planets,
        dynamics,
        disk=None,
        midplane_inc_deg=0.0,
        midplane_pa_deg=0.0,
        *,
        body_index=None,
    ):
        """Compose photometry and a fixed Jacobi hierarchy.

        ``body_index[j]`` is the zero-based chain planet for flattened scene
        planet j. It is structural metadata, never inferred during a solve.
        """
        self.star = star
        self.planets = tuple(planets)
        self.dynamics = dynamics
        self.disk = disk
        self.midplane_inc_deg = midplane_inc_deg
        self.midplane_pa_deg = midplane_pa_deg
        self.body_index = (
            tuple(range(self.n_planets)) if body_index is None else tuple(body_index)
        )

    def __check_init__(self):
        """Validate body counts and the static scene-to-chain permutation."""
        count = self.n_planets
        if self.dynamics.planets.mass_Msun.shape != (count,):
            raise ValueError("Photometric planet count must match dynamics bodies")
        if (
            len(self.body_index) != count
            or any(
                not isinstance(i, Integral) or isinstance(i, bool)
                for i in self.body_index
            )
            or sorted(self.body_index) != list(range(count))
        ):
            raise ValueError("body_index must be an integer permutation of the planets")

    @classmethod
    def from_classical(
        cls,
        star,
        planets,
        *,
        star_mass_Msun,
        planet_mass_Msun,
        dist_pc,
        a_AU,
        e,
        i_rad,
        W_rad,
        w_rad,
        M0_rad,
        epoch_d=0.0,
        epoch=_DEFAULT_EPOCH,
        order=None,
        disk=None,
        midplane_inc_deg=0.0,
        midplane_pa_deg=0.0,
    ):
        """Build a Jacobi chain from elements and masses in scene planet order.

        ``order`` lists scene planet indices in inner-to-outer chain order.
        With None, a stable host sort of concrete semi-major axes supplies it;
        traced axes require an explicit order. Unknown (nonfinite or
        nonpositive) planet masses become zero here. Scalars broadcast to the
        flattened planet count. ``epoch_d`` is the anomaly epoch in days;
        ``epoch`` is SkyFrame reference metadata. Observation ``time_jd``
        values are passed unchanged to orbix's ``time_d`` in this API.
        """
        planets = tuple(planets)
        count = sum(p.n_planets for p in planets)
        if order is None:
            if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(a_AU)):
                raise ValueError(
                    "Traced a_AU requires an explicit inner-to-outer order"
                )
            sizes = np.broadcast_to(np.asarray(a_AU), (count,))
            order = tuple(int(i) for i in np.argsort(sizes, kind="stable"))
        else:
            order = tuple(order)
        if (
            len(order) != count
            or any(not isinstance(i, Integral) or isinstance(i, bool) for i in order)
            or sorted(order) != list(range(count))
        ):
            raise ValueError("order must be an integer permutation of the planets")
        indices = jnp.asarray(order, dtype=int)
        elements = {
            name: jnp.broadcast_to(jnp.asarray(value), (count,))[indices]
            for name, value in dict(
                a_AU=a_AU,
                e=e,
                i_rad=i_rad,
                W_rad=W_rad,
                w_rad=w_rad,
                M0_rad=M0_rad,
                epoch_d=epoch_d,
            ).items()
        }
        masses = jnp.broadcast_to(_known_planet_mass(planet_mass_Msun), (count,))[
            indices
        ]
        dynamics = DynamicsSystem.from_classical(
            Body(star_mass_Msun), Body(masses), SkyFrame(dist_pc, epoch), **elements
        )
        inverse = tuple(order.index(i) for i in range(count))
        return cls(
            star,
            planets,
            dynamics,
            disk,
            midplane_inc_deg,
            midplane_pa_deg,
            body_index=inverse,
        )

    @property
    def dist_pc(self):
        """Observer distance [pc], owned by the dynamics sky frame."""
        return self.dynamics.frame.dist_pc

    @property
    def star_mass_Msun(self):
        """Stellar mass [solar masses], owned by the dynamics star."""
        return self.dynamics.star.mass_Msun

    @property
    def planet_mass_Msun(self):
        """Planet masses [solar masses] in flattened scene order."""
        return self.dynamics.planets.mass_Msun[jnp.asarray(self.body_index, dtype=int)]

    def ephemeris(self, time_jd: Array, *, solver=None) -> _SceneEphemeris:
        """Solve once with Markley by default and retain scene-order body indices.

        The returned orbix Ephemeris stores ``(K_total+1, T, 3)`` states
        in chain order. Its ``scene_planet_indices`` array addresses the
        planets in scene order. Pass this object as ``ephemeris=`` to any
        geometry consumer to reuse the solve; dates must match that solve.
        ``solver`` accepts an orbix.kepler strategy. Stored orbit rates follow
        orbix semantics: editing masses changes barycentric weights, not rates.
        """
        solved = self.dynamics.solve(
            time_jd, solver=Markley() if solver is None else solver
        )
        return _SceneEphemeris(
            solved.position_AU,
            solved.velocity_AU_per_d,
            self.dynamics.frame.dist_pc,
            jnp.asarray(self.body_index, dtype=int) + 1,
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
                dist_pc=self.dynamics.frame.dist_pc,
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
            lambda w: jax.vmap(
                lambda t: self.star.spec_flux_density(
                    w, t, dist_pc=self.dynamics.frame.dist_pc
                )
            )(time_jd),
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
            lambda w: jax.vmap(
                lambda t: self.star.spec_flux_density(
                    w, t, dist_pc=self.dynamics.frame.dist_pc
                )
            )(time_jd),
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
            indent("dynamics: " + repr(self.dynamics)),
            f"  body_index: {self.body_index}",
        ]
        for i, p in enumerate(self.planets):
            lines.append(indent(f"planet[{i}]:"))
            lines.append(indent(repr(p), prefix="    "))
        if self.disk is not None:
            lines.append(indent("disk: " + repr(self.disk)))
        else:
            lines.append("  disk: None")
        return "\n".join(lines)

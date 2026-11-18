"""Photometric scenes with one orbix system owning all dynamical state."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import contrast_to_delta_mag
from hwoutils.epoch import Epoch
from hwoutils.spectral import Transmission, bin_mean
from jaxtyping import Array
from orbix.kepler import Markley
from orbix.system import Body, Ephemeris, SkyFrame
from orbix.system import System as DynamicsSystem

from .._repr import indent
from ..disk import AbstractDisk
from . import _system_batch
from ._system_batch import _PAD_ORBIT
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


def _batch_method(method):
    """Apply an unbatched reader over each static leading scene axis."""
    parameters = signature(method)

    @wraps(method)
    def dispatch(self, *args, **kwargs):
        if not self.batch_shape:
            return method(self, *args, **kwargs)
        bound = parameters.bind(self, *args, **kwargs)
        bound.apply_defaults()
        options = dict(bound.arguments)
        options.pop("self")
        time = jnp.asarray(options.pop("time_d"))
        paired = time.ndim != 1
        if paired and time.shape[:-1] != self.batch_shape:
            raise ValueError("time_d must have shape (T,) or B + (T,)")
        has_ephemeris = "ephemeris" in options
        ephemeris = options.pop("ephemeris", None)

        def read(scene, dates, solved):
            extra = dict(ephemeris=solved) if has_ephemeris else {}
            return method(scene, time_d=dates, **options, **extra)

        scene_axes = jax.tree.map(
            lambda x: 0 if eqx.is_array(x) and x.ndim else None, self
        )
        for _ in self.batch_shape:
            read = jax.vmap(
                read,
                in_axes=(
                    scene_axes,
                    0 if paired else None,
                    0 if ephemeris is not None else None,
                ),
            )
        return read(self, time, ephemeris)

    dispatch.__doc__ = (
        method.__doc__
        + "\n\n    Batch axes B prefix outputs; times are (T,) or B + (T,)."
    )
    return dispatch


def _mask_dynamics(dynamics, chain_valid):
    """Replace inactive mass/orbit leaves before any orbital arithmetic."""
    orbit = jax.tree_util.tree_map_with_path(
        lambda path, x: jnp.where(
            chain_valid[:, None] if x.ndim == 2 else chain_valid,
            x,
            jnp.asarray(_PAD_ORBIT[path[-1].name], dtype=x.dtype),
        ),
        dynamics.orbits,
    )
    return eqx.tree_at(
        lambda d: (d.planets.mass_Msun, d.orbits),
        dynamics,
        (jnp.where(chain_valid, dynamics.planets.mass_Msun, 0.0), orbit),
    )


@eqx.filter_jit
def _masked_solve(dynamics, chain_valid, time_d, solver):
    """Fuse masking with the solve, including when called eagerly."""
    return _mask_dynamics(dynamics, chain_valid).solve(time_d, solver=solver)


class System(eqx.Module):
    """Astrophysical scene: star + tuple of planets + optional disk.

    All ``time_d`` arguments are plain arrays of days from the scene epoch,
    ``dynamics.frame.epoch``. Convert absolute dates on the host with
    ``system.epoch.days(...)`` before passing offsets to JAX.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        dynamics: Orbix Jacobi chain, declared inner to outer.
        epoch: Read-only reference to ``dynamics.frame.epoch``.
        valid: Boolean B + (K,) mask; defaults to all True. Batched scenes
            contain exactly one stacked Planet group and no disk. Every
            numeric leaf, including spectral grids, carries the batch axes.
        body_index: Static permutation mapping flattened scene planet indices to
            zero-based chain planet indices. Defaults to the identity.
        disk: Optional extended-source disk (``AbstractDisk | None``).
        midplane_inc_deg: Disk midplane inclination [deg] in [0, 180],
            with 0 face-on (the default). Together with ``midplane_position_angle_deg``,
            orients parametric disks and visualization outlines in the
            right-handed (east, north, away) sky frame. ``io.from_exovista``
            preserves the FITS star header's I. Planet orbits and loaded
            disk images already carry their own geometry.
        midplane_position_angle_deg: Major-axis position angle north through
            east [deg]. ExoVista headers map to (90 + PA) modulo 360.
    """

    star: AbstractStar
    planets: tuple[Planet, ...]
    dynamics: DynamicsSystem
    valid: Array
    _source_layout: Array | None = None
    disk: AbstractDisk | None = None
    midplane_inc_deg: float = 0.0
    midplane_position_angle_deg: float = 0.0
    body_index: tuple[int, ...] = eqx.field(static=True, default=())

    def photon_radiance(self, wavelength_nm, time_d, *, grid):
        """Disk photon radiance on grid, using the system's midplane angles.

        Returns ph s^-1 m^-2 nm^-1 sr^-1 with shape wavelength_nm.shape
        + grid.shape, measure="per_unit_area". No disk returns a zero image.
        Batched systems exclude disks.
        """
        from ..disk.base import check_sky_grid, radiance_image

        check_sky_grid(grid)
        east, north = grid.orientation.detector_to_sky(*grid.coordinates())
        return radiance_image(
            self.photon_radiance_at(wavelength_nm, time_d, east, north), grid
        )

    def photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec
    ) -> Array:
        """Disk photon radiance in photons/s/m^2/nm/sr at the given sky offsets.

        Axes of wavelength_nm precede the broadcast shape of the supplied
        east/north offsets. Positions are in arcseconds and time_d is scalar
        days from the scene epoch. No disk gives zero.
        """
        from ..disk.composite import orient_disk

        if self.disk is None:
            sky_shape = jnp.broadcast_shapes(
                jnp.shape(east_arcsec), jnp.shape(north_arcsec)
            )
            return jnp.zeros(jnp.shape(wavelength_nm) + sky_shape)
        disk = orient_disk(
            self.disk, self.midplane_inc_deg, self.midplane_position_angle_deg
        )
        return disk.photon_radiance_at(
            wavelength_nm,
            time_d,
            east_arcsec,
            north_arcsec,
            star=self.star,
            dist_pc=self.dist_pc,
        )

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
        midplane_position_angle_deg=0.0,
        *,
        body_index=None,
        valid=None,
    ):
        """Compose photometry and a fixed Jacobi hierarchy.

        ``body_index[j]`` is the zero-based chain planet for flattened scene
        planet j. It is structural metadata, never inferred during a solve.
        """
        self.star = star
        self._source_layout = None
        self.planets = (planets,) if isinstance(planets, Planet) else tuple(planets)
        self.dynamics = dynamics
        shape = dynamics.planets.mass_Msun.shape
        self.valid = (
            jnp.ones(shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        self.disk = disk
        self.midplane_inc_deg = midplane_inc_deg
        self.midplane_position_angle_deg = midplane_position_angle_deg
        self.body_index = (
            tuple(range(self.n_planets)) if body_index is None else tuple(body_index)
        )

    def __check_init__(self):
        """Validate body counts and the static scene-to-chain permutation."""
        count = self.n_planets
        if self.dynamics.planets.mass_Msun.shape != (*self.batch_shape, count):
            raise ValueError("Photometric planet count must match dynamics bodies")
        if self.valid.shape != (*self.batch_shape, count):
            raise ValueError("valid must have shape B + (K,)")
        if self.batch_shape and (len(self.planets) != 1 or self.disk is not None):
            raise ValueError(
                "Batched scenes require one stacked Planet group and no disk"
            )
        if (
            len(self.body_index) != count
            or any(
                not isinstance(i, Integral) or isinstance(i, bool)
                for i in self.body_index
            )
            or sorted(self.body_index) != list(range(count))
        ):
            raise ValueError("body_index must be an integer permutation of the planets")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading scene axes, determined statically from stellar mass shape."""
        return self.dynamics.star.mass_Msun.shape

    @property
    def capacity(self) -> int:
        """Static number of planet slots, including the padded tail."""
        return self.dynamics.planets.mass_Msun.shape[-1]

    @classmethod
    def from_systems(cls, systems, *, capacity=None, batch_shape=None):
        """Pack same-class scenes into NumPy float64 leaves, sorted inner to outer.

        The default batch shape is (len(systems),); capacity defaults to the
        largest valid count. Disks, mixed epochs and mixed model classes raise.
        At least one Planet group must supply the physical-model class; an
        empty group is sufficient. Host layout data restores source group
        boundaries and scene order when unpacking.
        """
        return _system_batch.from_systems(
            cls, systems, capacity=capacity, batch_shape=batch_shape
        )

    def to_systems(self):
        """Return unpadded host scenes in row-major batch order."""
        return _system_batch.to_systems(self)

    def to_table(self) -> dict[str, np.ndarray]:
        """Return all leaves by JAX key path, plus scalar K and epoch parts."""
        return _system_batch.to_table(self)

    @classmethod
    def from_table(cls, table, *, template):
        """Restore a table; template supplies model classes and static metadata."""
        return _system_batch.from_table(cls, table, template=template)

    def take(self, index):
        """Gather a scalar index along the last batch axis, under jit or vmap."""
        if not self.batch_shape:
            raise ValueError("take requires at least one batch axis")
        axis = len(self.batch_shape) - 1
        return jax.tree.map(
            lambda x: (
                jnp.take(x, index, axis=axis) if eqx.is_array(x) and x.ndim else x
            ),
            self,
        )

    def _chain_valid(self):
        inverse = tuple(self.body_index.index(i) for i in range(self.capacity))
        return self.valid[jnp.asarray(inverse, dtype=int)]

    def _masked_dynamics(self):
        """Remove every padded dynamical derivative before the orbital solve."""
        return _mask_dynamics(self.dynamics, self._chain_valid())

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
        valid=None,
        disk=None,
        midplane_inc_deg=0.0,
        midplane_position_angle_deg=0.0,
    ):
        """Build a Jacobi chain from elements and masses in scene planet order.

        ``order`` lists scene planet indices in inner-to-outer chain order.
        With None, a stable host sort of concrete semi-major axes supplies it;
        traced axes require an explicit order. Unknown (nonfinite or
        nonpositive) planet masses become zero here. Scalars broadcast to the
        flattened planet count. ``epoch_d`` and observation ``time_d`` are
        days from the scene epoch. ``epoch`` is host-side SkyFrame metadata;
        convert absolute dates once with ``system.epoch.days(...)`` before
        transferring the resulting offsets to JAX.
        """
        planets = (planets,) if isinstance(planets, Planet) else tuple(planets)
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
            midplane_position_angle_deg,
            body_index=inverse,
            valid=valid,
        )

    @property
    def dist_pc(self):
        """Observer distance [pc], owned by the dynamics sky frame."""
        return self.dynamics.frame.dist_pc

    @property
    def epoch(self) -> Epoch:
        """Host-side scene epoch, owned by the dynamics sky frame."""
        return self.dynamics.frame.epoch

    @property
    def star_mass_Msun(self):
        """Stellar mass [solar masses], owned by the dynamics star."""
        return self.dynamics.star.mass_Msun

    @property
    def planet_mass_Msun(self):
        """Planet masses [solar masses] in flattened scene order."""
        return self.dynamics.planets.mass_Msun[
            ..., jnp.asarray(self.body_index, dtype=int)
        ]

    @_batch_method
    def ephemeris(self, time_d: Array, *, solver=None) -> _SceneEphemeris:
        """Solve once with Markley by default and retain scene-order body indices.

        ``time_d`` is days from the scene epoch.

        The returned orbix Ephemeris stores ``(K_total+1, T, 3)`` states
        in chain order. Its ``scene_planet_indices`` array addresses the
        planets in scene order. Pass this object as ``ephemeris=`` to any
        geometry consumer to reuse the solve; dates must match that solve.
        ``solver`` accepts an orbix.kepler strategy. Stored orbit rates follow
        orbix semantics: editing masses changes barycentric weights, not rates.
        """
        solved = _masked_solve(
            self.dynamics,
            self._chain_valid(),
            time_d,
            Markley() if solver is None else solver,
        )
        return _SceneEphemeris(
            solved.position_AU,
            solved.velocity_AU_per_d,
            self.dynamics.frame.dist_pc,
            jnp.asarray(self.body_index, dtype=int) + 1,
        )

    @_batch_method
    def positions(self, time_d: Array, *, ephemeris=None) -> Array:
        """Star-relative on-sky positions, shape ``(2, K_total, T)``.

        ``time_d`` is days from the scene epoch.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(eph.scene_planet_indices, 0), axis=0)

    @_batch_method
    def position_angles(self, time_d: Array, *, ephemeris=None) -> Array:
        """Position angles east of north in ``[0, 360)``, ``(K_total, T)``.

        ``time_d`` is days from the scene epoch.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return eph.position_angle_deg(eph.scene_planet_indices, 0)

    @_batch_method
    def star_reflex(self, time_d: Array, *, ephemeris=None) -> Array:
        """Stellar (east, north) barycentric offsets [arcsec], ``(2, T)``.

        ``time_d`` is days from the scene epoch.

        The Jacobi chain includes interior planets in each orbit's reference
        mass and centre; it is a Kepler hierarchy, without mutual perturbations.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(0, "barycenter"), axis=0)

    @_batch_method
    def star_radial_velocity(self, time_d: Array, *, ephemeris=None) -> Array:
        """Stellar barycentric radial velocity [m/s], positive receding, ``(T,)``.

        ``time_d`` is days from the scene epoch.

        Read the Jacobi ephemeris without adding a systemic velocity.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return eph.radial_velocity_m_per_s(0, "barycenter")

    def _photometry(self, wavelength_nm, time_d, ephemeris, *, flux=False):
        """Dispatch physical models on star-relative phase/distance, each (K, T).

        ``time_d`` is days from the scene epoch.
        """
        if jnp.size(wavelength_nm) != 1:
            raise ValueError("need one wavelength; use contrast_cube")
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        phase = eph.phase_angle_rad(eph.scene_planet_indices, 0)
        distance = eph.distance_AU(eph.scene_planet_indices, 0)
        values = []
        start = 0
        for planet in self.planets:
            stop = start + planet.n_planets
            geometry = dict(
                phase_angle_rad=phase[start:stop],
                star_planet_distance_AU=distance[start:stop],
                Rp_Rearth=jnp.where(self.valid[start:stop], planet.Rp_Rearth, 0.0),
                time_d=time_d,
                star=self.star,
                dist_pc=self.dynamics.frame.dist_pc,
            )
            mask = self.valid[start:stop]
            model = jax.tree_util.tree_map_with_path(
                lambda path, x, mask=mask, planet=planet: (
                    jnp.where(
                        mask.reshape((mask.size,) + (1,) * (jnp.ndim(x) - 1)),
                        x,
                        jax.lax.stop_gradient(x),
                    )
                    if _system_batch._per_planet(path, x, planet.n_planets)
                    else x
                ),
                planet.physical_model,
            )
            values.append(
                model.photon_flux_density(wavelength_nm, **geometry)
                if flux
                else model.contrast_at_time(wavelength_nm=wavelength_nm, **geometry)
            )
            start = stop
        result = (
            jnp.concatenate(values, axis=0) if values else jnp.empty((0, time_d.size))
        )
        return jnp.where(self.valid[:, None], result, 0.0) * self.valid[:, None]

    @_batch_method
    def contrasts(
        self, wavelength_nm: Array, time_d: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet contrast at a scalar wavelength [nm], shape ``(K_total, T)``.

        ``time_d`` is days from the scene epoch.
        """
        return self._photometry(wavelength_nm, time_d, ephemeris)

    @_batch_method
    def contrast_cube(
        self, wavelengths_nm: Array, time_d: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet contrast at each wavelength, shape ``(W, K_total, T)``.

        Args:
            wavelengths_nm: Wavelengths [nm], shape ``(W,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional scene ephemeris solved at these dates.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jax.vmap(lambda w: self.contrasts(w, time_d, ephemeris=eph))(
            jnp.asarray(wavelengths_nm)
        )

    @_batch_method
    def bin_mean_contrast_cube(
        self,
        centers_nm: Array,
        widths_nm: Array,
        time_d: Array,
        *,
        n_nodes: int = 4,
        ephemeris=None,
    ) -> Array:
        """Ratio of bin-integrated planet and star photons, ``(W, K_total, T)``.

        ``time_d`` is days from the scene epoch.

        Centers and full nonnegative widths are ``(W,)`` in nm; dates are
        ``(T,)``. Zero width returns ``contrast_cube``. For linear stellar
        tables each stellar interval is integrated with ``n_nodes`` Gauss-
        Legendre nodes, clipping the end intervals to the bin. This resolves
        stellar lines and integrates the product of stellar flux and planet
        contrast, with an exact stellar denominator for linear photon tables.
        Resolve additional planetary features by increasing ``n_nodes``. Analytic
        stars use one Gauss-Legendre rule per bin. This is a flux-weighted contrast,
        not a mean of monochromatic contrasts. See Davis and Rabinowitz (1984),
        Methods of Numerical Integration, for the quadrature rule. The static
        interval cap covers all stellar knots.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        knots = self.star.quadrature_knots_nm
        numerator = bin_mean(
            lambda w: self.planet_flux_densities(w, time_d, ephemeris=eph),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=None if knots is None else len(knots) + 1,
        )
        denominator = bin_mean(
            lambda w: jax.vmap(
                lambda t: self.star.photon_flux_density(
                    w, t, dist_pc=self.dynamics.frame.dist_pc
                )
            )(time_d),
            centers_nm,
            widths_nm,
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=None if knots is None else len(knots) + 1,
        )
        centers, widths = jnp.asarray(centers_nm), jnp.asarray(widths_nm)
        point_bin = centers - widths / 2 == centers + widths / 2
        safe_denominator = jnp.where(point_bin[:, None], 1.0, denominator)
        ratio = numerator / safe_denominator[:, None, :]
        return jnp.where(
            point_bin[:, None, None],
            self.contrast_cube(centers, time_d, ephemeris=eph),
            ratio,
        )

    @_batch_method
    def band_contrasts(
        self, bandpass: Transmission, time_d: Array, *, n_nodes: int = 4, ephemeris=None
    ) -> Array:
        """Ratio of band-integrated planet and star photons, ``(K_total, T)``.

        ``time_d`` is days from the scene epoch.

        Multiply photon flux densities by the photon-counting transmission.
        Split quadrature at both the bandpass table and stellar spectral
        knots. ``n_nodes`` is the Gauss-Legendre order per interval: linear
        transmission is exact at order one, its product with a linear photon
        spectrum at order two. Increase the order to resolve smooth spectral
        curvature or additional planetary features. The integrated stellar
        photon flux must be nonzero.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        centers = jnp.atleast_1d((bandpass.low_nm + bandpass.high_nm) / 2)
        widths = jnp.atleast_1d(bandpass.high_nm - bandpass.low_nm)
        knots = bandpass.quadrature_knots_nm
        if self.star.quadrature_knots_nm is not None:
            knots = jnp.sort(jnp.concatenate((knots, self.star.quadrature_knots_nm)))
        quadrature = dict(
            n_nodes=n_nodes,
            knots_nm=knots,
            max_knots=len(knots) + 1,
            weight_fn=bandpass,
        )
        numerator = bin_mean(
            lambda w: self.planet_flux_densities(w, time_d, ephemeris=eph),
            centers,
            widths,
            **quadrature,
        )[0]
        denominator = bin_mean(
            lambda w: jax.vmap(
                lambda t: self.star.photon_flux_density(
                    w, t, dist_pc=self.dynamics.frame.dist_pc
                )
            )(time_d),
            centers,
            widths,
            **quadrature,
        )[0]
        return numerator / denominator[None, :]

    @_batch_method
    def planet_flux_densities(
        self, wavelength_nm: Array, time_d: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet flux density [ph/s/m^2/nm], shape ``(K_total, T)``.

        ``time_d`` is days from the scene epoch.
        """
        return self._photometry(wavelength_nm, time_d, ephemeris, flux=True)

    @_batch_method
    def alpha_dMag(
        self, time_d: Array, *, wavelength_nm: float = 600.0, ephemeris=None
    ) -> tuple[Array, Array]:
        """Per-planet separation + dMag at wavelength_nm (default 600 nm).

        ``time_d`` is days from the scene epoch.

        Each result has shape ``(K_total, T)``; wavelength_nm is scalar.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        alpha = eph.separation_arcsec(eph.scene_planet_indices, 0)
        contrast = self.contrasts(jnp.asarray(wavelength_nm), time_d, ephemeris=eph)
        safe = jnp.where(self.valid[:, None], contrast, 1.0)
        dMag = contrast_to_delta_mag(safe + jnp.finfo(contrast.dtype).tiny)
        return alpha, dMag

    def __repr__(self) -> str:
        """Tree-shaped summary: star + planets + disk + midplane geometry."""
        if self.batch_shape:
            return (
                f"System(batch_shape={self.batch_shape}, capacity={self.capacity}, "
                f"star={type(self.star).__name__}, "
                f"model={type(self.planets[0].physical_model).__name__})"
            )
        n_modules = len(self.planets)
        n_total = self.n_planets
        lines = [
            (
                f"System(n_planet_modules={n_modules}, "
                f"K_total={n_total}, "
                f"midplane: i={self.midplane_inc_deg:.2f} deg, "
                f"PA={self.midplane_position_angle_deg:.2f} deg)"
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

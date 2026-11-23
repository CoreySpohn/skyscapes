"""The Scene: star, planets, disk and local zodi, over one orbix system."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import contrast_to_delta_mag
from hwoutils.epoch import Epoch
from jaxtyping import Array
from optixstuff.spectral import Quadrature, Resolution, spectral_breaks
from orbix.kepler import Markley
from orbix.system import Body, Ephemeris, SkyFrame
from orbix.system import System as DynamicsSystem

from .._repr import indent
from .._tables import ProductTable, is_table
from ..disk import AbstractDisk
from . import _system_batch
from ._system_batch import _PAD_ORBIT
from .local_zodi import AbstractLocalZodi
from .planet import Planet
from .sources import Sources
from .star import AbstractStar

_DEFAULT_EPOCH = Epoch()


class _SceneEphemeris(Ephemeris):
    """Chain-order states with body indices for planets in scene order.

    ``scene_planet_indices`` has shape ``(K_total,)`` and indexes the body
    axis of all inherited observables; body zero is always the star.

    Attributes:
        scene_planet_indices: Integer indices of the scene's planets on the body
            axis, in scene order, shape ``(K_total,)``.
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


class Scene(eqx.Module):
    """Everything in the field of view: star, planets, disk and local zodi.

    All ``time_d`` arguments are plain arrays of days from the scene epoch,
    ``dynamics.frame.epoch``. Convert absolute dates on the host with
    ``scene.epoch.days(jd=...)`` before passing offsets to JAX. The read-only
    ``epoch`` property returns ``dynamics.frame.epoch``.

    Attributes:
        star: Host star (``AbstractStar``).
        planets: Variable-length tuple of ``Planet``.
        dynamics: Orbix Jacobi chain, declared inner to outer.
        valid: Boolean B + (K,) mask; defaults to all True. Batched scenes
            contain exactly one stacked Planet group and no disk. Every
            numeric leaf, including spectral grids, carries the batch axes.
        disk: Optional extended source, the only exozodi (``AbstractDisk | None``):
            an ExoVista disk, a parametric disk, or an analytic exozodi model
            such as ``skyscapes.disk.AYOExozodi``.
        local_zodi: Optional local (solar-system) zodiacal light, any
            :class:`~skyscapes.scene.AbstractLocalZodi` model. Unbatched scenes only.
        midplane_inc_deg: Disk midplane inclination [deg] in [0, 180],
            with 0 face-on (the default). Together with ``midplane_position_angle_deg``,
            orients parametric disks and visualization outlines in the
            right-handed (east, north, away) sky frame. ``io.from_exovista``
            preserves the FITS star header's I. Planet orbits and loaded
            disk images already carry their own geometry.
        midplane_position_angle_deg: Major-axis position angle north through
            east [deg]. ExoVista headers map to (90 + PA) modulo 360.
        body_index: Static permutation mapping flattened scene planet indices to
            zero-based chain planet indices. Defaults to the identity.
    """

    star: AbstractStar
    planets: tuple[Planet, ...]
    dynamics: DynamicsSystem
    valid: Array
    _source_layout: Array | None = None
    disk: AbstractDisk | None = None
    local_zodi: AbstractLocalZodi | None = None
    midplane_inc_deg: float = 0.0
    midplane_position_angle_deg: float = 0.0
    body_index: tuple[int, ...] = eqx.field(static=True, default=())

    def photon_radiance(self, wavelength_nm, time_d, *, grid, star=None):
        """Disk photon radiance on grid, using the scene's midplane angles.

        Returns ph s^-1 m^-2 nm^-1 sr^-1 with shape wavelength_nm.shape
        + grid.shape. A scene without a disk returns zero radiance.
        Batched scenes exclude disks and local zodi.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Observation time, scalar days from the scene epoch.
            grid: Sky pixel grid on which the radiance is sampled.
            star: Star the disk is rendered against; None uses the scene's star.
                A band-rule route passes ``Sources.node_star(w, q)``, so a disk
                defined relative to the star carries the star's node-effective
                value instead of its point sample.

        Returns:
            Disk photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape
            ``wavelength_nm.shape + grid.shape``.
        """
        from ..disk.base import check_sky_grid, radiance_image

        check_sky_grid(grid)
        east, north = grid.orientation.detector_to_sky(*grid.coordinates())
        return radiance_image(
            self.photon_radiance_at(wavelength_nm, time_d, east, north, star=star),
            grid,
        )

    def photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star=None
    ) -> Array:
        """Disk photon radiance in photons/s/m^2/nm/sr at the given sky offsets.

        Axes of wavelength_nm precede the broadcast shape of the supplied
        east/north offsets. Positions are in arcseconds and time_d is scalar
        days from the scene epoch. No disk gives zero.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Observation time, scalar days from the scene epoch.
            east_arcsec: Sky offsets east of the star [arcsec].
            north_arcsec: Sky offsets north of the star [arcsec], broadcast with
                ``east_arcsec``.
            star: Star the disk is rendered against; None uses the scene's star.
                A read vectorised over a band rule's nodes passes
                ``Sources.node_star()`` with ``wavelength_nm`` the rule's nodes.

        Returns:
            Disk photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape
            ``wavelength_nm.shape`` followed by the broadcast offset shape.
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
            star=self.star if star is None else star,
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
        local_zodi=None,
        body_index=None,
        valid=None,
    ):
        """Compose photometry and a fixed Jacobi hierarchy.

        ``body_index[j]`` is the zero-based chain planet for flattened scene
        planet j. It is structural metadata, never inferred during a solve.

        Args:
            star: Host star.
            planets: One ``Planet`` or a sequence of them, in scene order.
            dynamics: Orbix Jacobi chain of the star and planets, inner to outer.
            disk: Optional extended source, or None.
            midplane_inc_deg: Disk midplane inclination [deg], 0 face-on.
            midplane_position_angle_deg: Disk major-axis position angle north
                through east [deg].
            local_zodi: Optional local zodiacal light model, or None.
            body_index: Chain planet index for each flattened scene planet; None
                uses the identity.
            valid: Boolean mask of real planet slots, shape ``B + (K,)``; None
                marks all valid.
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
        self.local_zodi = local_zodi
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
        if self.batch_shape and (
            len(self.planets) != 1
            or self.disk is not None
            or self.local_zodi is not None
        ):
            raise ValueError(
                "Batched scenes require one stacked Planet group, no disk and no "
                "local zodi"
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
    def from_scenes(cls, scenes, *, capacity=None, batch_shape=None):
        """Pack same-class scenes into NumPy float64 leaves, sorted inner to outer.

        The default batch shape is (len(scenes),); capacity defaults to the
        largest valid count. Disks, mixed epochs and mixed model classes raise.
        At least one Planet group must supply the physical-model class; an
        empty group is sufficient. Host layout data restores source group
        boundaries and scene order when unpacking.

        Args:
            scenes: Unbatched scenes to pack.
            capacity: Planet slots per scene; None uses the largest valid count.
            batch_shape: Leading batch axes; None uses ``(len(scenes),)``.

        Returns:
            A batched scene on the host with NumPy float64 leaves.
        """
        return _system_batch.from_scenes(
            cls, scenes, capacity=capacity, batch_shape=batch_shape
        )

    def to_scenes(self):
        """Return unpadded host scenes in row-major batch order.

        Returns:
            A list of unbatched host scenes.
        """
        return _system_batch.to_scenes(self)

    def to_table(self) -> dict[str, np.ndarray]:
        """Return all leaves by JAX key path, plus scalar K and epoch parts.

        Returns:
            A dict of host arrays keyed by leaf path, plus ``body_index``,
            ``K`` and the epoch entries.
        """
        return _system_batch.to_table(self)

    @classmethod
    def from_table(cls, table, *, template):
        """Restore a table; template supplies model classes and static metadata.

        Args:
            table: A table written by :meth:`to_table`.
            template: A scene supplying model classes and static metadata.

        Returns:
            The restored scene on the host.
        """
        return _system_batch.from_table(cls, table, template=template)

    def take(self, index):
        """Gather a scalar index along the last batch axis, under jit or vmap.

        Args:
            index: Scalar integer index along the last batch axis.

        Returns:
            A scene with that batch axis removed.
        """
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
        ra_deg=None,
        dec_deg=None,
        order=None,
        valid=None,
        disk=None,
        midplane_inc_deg=0.0,
        midplane_position_angle_deg=0.0,
        local_zodi=None,
    ):
        """Build a Jacobi chain from elements and masses in scene planet order.

        ``order`` lists scene planet indices in inner-to-outer chain order.
        With None, a stable host sort of concrete semi-major axes supplies it;
        traced axes require an explicit order. Unknown (nonfinite or
        nonpositive) planet masses become zero here. Scalars broadcast to the
        flattened planet count. ``epoch_d`` and observation ``time_d`` are
        days from the scene epoch. ``epoch`` is host-side SkyFrame metadata;
        convert absolute dates once with ``scene.epoch.days(jd=...)`` before
        transferring the resulting offsets to JAX. ``ra_deg`` and ``dec_deg``
        place the scene on the sky (ICRS degrees, both or neither); without
        them the scene is unplaced.

        Args:
            star: Host star.
            planets: One ``Planet`` or a sequence of them, in scene order.
            star_mass_Msun: Stellar mass [solar masses].
            planet_mass_Msun: Planet masses [solar masses]; a scalar or shape
                ``(K_total,)`` in scene order.
            dist_pc: Distance to the system [pc].
            a_AU: Semi-major axes [AU]; a scalar or shape ``(K_total,)``.
            e: Eccentricities, dimensionless; a scalar or shape ``(K_total,)``.
            i_rad: Inclinations [rad]; a scalar or shape ``(K_total,)``.
            W_rad: Longitudes of the ascending node [rad]; a scalar or shape
                ``(K_total,)``.
            w_rad: Arguments of periapsis [rad]; a scalar or shape
                ``(K_total,)``.
            M0_rad: Mean anomalies at ``epoch_d`` [rad]; a scalar or shape
                ``(K_total,)``.
            epoch_d: Epoch of the elements, days from the scene epoch.
            epoch: Scene reference epoch, host-side metadata.
            ra_deg: Right ascension [deg, ICRS], or None for an unplaced scene.
            dec_deg: Declination [deg, ICRS], or None for an unplaced scene.
            order: Scene planet indices in inner-to-outer chain order; None
                sorts by semi-major axis.
            valid: Boolean mask of real planet slots; None marks all valid.
            disk: Optional extended source, or None.
            midplane_inc_deg: Disk midplane inclination [deg], 0 face-on.
            midplane_position_angle_deg: Disk major-axis position angle north
                through east [deg].
            local_zodi: Optional local zodiacal light model, or None.

        Returns:
            The assembled scene.
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
            Body(star_mass_Msun),
            Body(masses),
            SkyFrame(dist_pc, epoch, ra_deg, dec_deg),
            **elements,
        )
        inverse = tuple(order.index(i) for i in range(count))
        return cls(
            star,
            planets,
            dynamics,
            disk,
            midplane_inc_deg,
            midplane_position_angle_deg,
            local_zodi=local_zodi,
            body_index=inverse,
            valid=valid,
        )

    @property
    def sky_frame(self) -> SkyFrame:
        """Where and when the scene is seen, owned by the dynamics."""
        return self.dynamics.frame

    @property
    def dist_pc(self):
        """Observer distance [pc], owned by the dynamics sky frame."""
        return self.dynamics.frame.dist_pc

    @property
    def epoch(self) -> Epoch:
        """Host-side scene epoch, owned by the dynamics sky frame."""
        return self.dynamics.frame.epoch

    @property
    def ra_deg(self):
        """Right ascension [deg, ICRS], or None for an unplaced scene."""
        return self.dynamics.frame.ra_deg

    @property
    def dec_deg(self):
        """Declination [deg, ICRS], or None for an unplaced scene."""
        return self.dynamics.frame.dec_deg

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

        Args:
            time_d: Observation times, days from the scene epoch.
            solver: Optional ``orbix.kepler`` strategy; None uses Markley.

        Returns:
            The solved ephemeris, with ``scene_planet_indices`` in scene order.
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
        """Star-relative (east, north) offsets in arcsec, north up, ``(2, K_total, T)``.

        ``time_d`` is days from the scene epoch.

        Args:
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Offsets [arcsec] of each planet from the star, shape ``(2, K_total, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(eph.scene_planet_indices, 0), axis=0)

    @_batch_method
    def position_angles(self, time_d: Array, *, ephemeris=None) -> Array:
        """Position angles east of north in ``[0, 360)``, ``(K_total, T)``.

        ``time_d`` is days from the scene epoch.

        Args:
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Position angles [deg], shape ``(K_total, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return eph.position_angle_deg(eph.scene_planet_indices, 0)

    @_batch_method
    def star_reflex(self, time_d: Array, *, ephemeris=None) -> Array:
        """Stellar (east, north) barycentric offsets [arcsec], ``(2, T)``.

        ``time_d`` is days from the scene epoch.

        The Jacobi chain includes interior planets in each orbit's reference
        mass and centre; it is a Kepler hierarchy, without mutual perturbations.

        Args:
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Stellar offsets (east, north) [arcsec], shape ``(2, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jnp.stack(eph.offset_arcsec(0, "barycenter"), axis=0)

    @_batch_method
    def star_radial_velocity(self, time_d: Array, *, ephemeris=None) -> Array:
        """Stellar barycentric radial velocity [m/s], positive receding, ``(T,)``.

        ``time_d`` is days from the scene epoch.

        Read the Jacobi ephemeris without adding a systemic velocity.

        Args:
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Radial velocity [m/s], shape ``(T,)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return eph.radial_velocity_m_per_s(0, "barycenter")

    def _planet_groups(self, time_d, ephemeris):
        """Yield each planet group's masked model, geometry and scale.

        Geometry is star-relative phase and distance, each (K, T), with the
        radius, times, host star and distance. ``time_d`` is days from the
        scene epoch.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        phase = eph.phase_angle_rad(eph.scene_planet_indices, 0)
        distance = eph.distance_AU(eph.scene_planet_indices, 0)
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
            # The planet's own scale, applied once by the caller, beside the radius.
            yield model, geometry, planet.flux_scale
            start = stop

    def _photometry(self, wavelength_nm, time_d, ephemeris, *, flux=False):
        """Dispatch physical models on star-relative phase/distance, each (K, T).

        ``time_d`` is days from the scene epoch.
        """
        if jnp.size(wavelength_nm) != 1:
            raise ValueError("need one wavelength; use contrast_cube")
        values = []
        for model, geometry, scale in self._planet_groups(time_d, ephemeris):
            value = (
                model.photon_flux_density(wavelength_nm, **geometry)
                if flux
                else model.contrast_at_time(wavelength_nm=wavelength_nm, **geometry)
            )
            values.append(value * scale.reshape(scale.shape + (1,) * (value.ndim - 1)))
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

        A point read: right for a monochromatic question and for a smooth
        contrast. A band quantity under a rough star (a thermal planet's
        contrast, any band integral) comes from :meth:`sources`.

        Args:
            wavelength_nm: A single wavelength [nm].
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Planet-to-star flux ratio, dimensionless, shape ``(K_total, T)``.
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

        Returns:
            Planet-to-star flux ratio, dimensionless, shape ``(W, K_total, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        return jax.vmap(lambda w: self.contrasts(w, time_d, ephemeris=eph))(
            jnp.asarray(wavelengths_nm)
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Union of stellar, planetary, disk and local-zodi wavelength breaks [nm].

        Returns:
            Sorted breakpoint wavelengths [nm] on the host.
        """
        models = (self.star, *self.planets)
        if self.disk is not None:
            models = (*models, self.disk)
        if self.local_zodi is not None:
            models = (*models, self.local_zodi)
        return spectral_breaks(*models)

    def spectral_tables(self) -> tuple:
        """Return the tabulated sources a band rule needs operators for, on the host.

        Pass them to ``optixstuff.spectral.Resolution.build`` with the rule:
        the star when it is a table, one product per planet group whose model
        declares knots (its table times the star's, on the merged knots), and
        the local zodi when it is a table. Each digest is checked against its
        concrete knots, so a knot grid replaced after construction is refused
        here with the remedy.

        Returns:
            The ``SpectralTable`` objects, in scene order.
        """
        tables = []
        if is_table(self.star):
            tables.append(self.star)
        for planet in self.planets:
            if planet.physical_model._tabulated:
                tables.append(ProductTable(planet.physical_model, self.star))
        if self.local_zodi is not None and is_table(self.local_zodi):
            tables.append(self.local_zodi)
        return tuple(tables)

    @_batch_method
    def sources(
        self,
        quadrature: Quadrature,
        resolution: Resolution,
        time_d: Array,
        *,
        ephemeris=None,
    ) -> Sources:
        """Read every source once on the nodes of a band rule.

        The star, each planet and the local zodi return node-effective values
        through their own node methods: summed with the rule's weights and a
        response at the nodes, each gives the band integral of the source
        times the response, exact in a tabulated spectrum and to the rule's
        degree in the response. Build the rule and the resolution on the host,
        ``Resolution.build(quadrature, *scene.spectral_tables())``; a tabulated
        source without a rule raises at trace time. Times are days from the
        scene epoch.

        Args:
            quadrature: Band rule from ``optixstuff.spectral.Quadrature.build``.
            resolution: Operators for the scene's tables on ``quadrature``.
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            The node-effective values of every source.
        """
        time_d = jnp.atleast_1d(jnp.asarray(time_d))
        dist_pc = self.dynamics.frame.dist_pc
        unscaled = self.star._node_flux_density(
            quadrature, resolution, time_d, dist_pc=dist_pc
        )
        scale = jnp.asarray(self.star.flux_scale)
        star = scale * unscaled
        values = []
        for model, geometry, planet_scale in self._planet_groups(time_d, ephemeris):
            value = model.node_photon_flux_density(
                quadrature, resolution, **geometry, star_nodes=star
            )
            values.append(value * planet_scale[:, None])
        nodes = quadrature.nodes_nm
        planets = (
            jnp.concatenate(values, axis=2)
            if values
            else jnp.zeros((*nodes.shape, 0, time_d.size))
        )
        valid = self.valid[:, None]
        planets = jnp.where(valid, planets, 0.0) * valid
        zodi = None
        if self.local_zodi is not None:
            zodi = jax.vmap(
                lambda t: self.local_zodi.node_radiance(
                    quadrature, resolution, self.sky_frame, t
                ),
                out_axes=-1,
            )(time_d)
        luminosity = getattr(self.star, "luminosity_Lsun", None)
        return Sources(
            unscaled_star=unscaled,
            star_scale=scale,
            planets=planets,
            local_zodi=zodi,
            star_diameter_arcsec=jnp.asarray(
                self.star.angular_diameter_arcsec(dist_pc)
            ),
            star_luminosity_Lsun=None
            if luminosity is None
            else jnp.asarray(luminosity),
        )

    @_batch_method
    def band_contrasts(
        self,
        quadrature: Quadrature,
        time_d: Array,
        *,
        transmission: Callable | None = None,
        resolution: Resolution | None = None,
        ephemeris=None,
    ) -> Array:
        """Ratio of integrated planet and star photons, shape ``(W, K_total, T)``.

        Build the quadrature on the host with this system and any tabulated
        transmission. The optional photon-counting response is evaluated on all
        ``(W, Q)`` wavelength nodes and must broadcast to that shape. Each
        finite-width band must transmit nonzero stellar flux. A monochromatic
        channel returns the model contrast directly, including for a reflected
        spectrum at a stellar zero. Times are days from the scene epoch.

        With ``resolution`` the sources are read once through :meth:`sources`,
        so a tabulated star or planet is integrated exactly without declaring
        its knots as breaks, and every band must lie inside every source
        table (``Resolution.build`` refuses one that leaves it). Without it the
        sources are sampled at the nodes: exact only when the quadrature splits
        at their knots. Spectra may then be undefined where transmission is
        zero, and an opaque response has undefined contrast.

        Args:
            quadrature: Host-built quadrature giving the band nodes [nm] and
                weights.
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            transmission: Optional photon-counting response evaluated on the
                ``(W, Q)`` node wavelengths [nm]; it must broadcast to that
                shape.
            resolution: Operators for the scene's tables on ``quadrature``, from
                ``Resolution.build(quadrature, *scene.spectral_tables())``; None
                samples the sources at the nodes.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Band-integrated planet-to-star photon ratio, dimensionless, shape
            ``(W, K_total, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        wave = quadrature.nodes_nm
        if len(quadrature.bands) == 1 and quadrature.bands[0].width_nm == 0:
            contrast = self.contrast_cube(wave[:, 0], time_d, ephemeris=eph)
            if transmission is None:
                return contrast
            response = jnp.broadcast_to(transmission(wave), wave.shape)[0, 0]
            return jnp.where(response > 0, contrast, jnp.nan)

        if resolution is not None:
            read = self.sources(quadrature, resolution, time_d, ephemeris=eph)
            planet, star = read.planets, read.star
            if transmission is not None:
                throughput = jnp.broadcast_to(transmission(wave), wave.shape)
                planet = planet * throughput[..., None, None]
                star = star * throughput[..., None]
            return quadrature.integrate(planet) / quadrature.integrate(star)[:, None, :]

        def densities(nodes):
            planet = jax.vmap(
                jax.vmap(lambda w: self.planet_flux_densities(w, time_d, ephemeris=eph))
            )(nodes)
            star = self.star.photon_flux_density(
                nodes[..., None], time_d, dist_pc=self.dynamics.frame.dist_pc
            )
            return planet, star

        if transmission is not None:
            throughput = jnp.broadcast_to(transmission(wave), wave.shape)

            def opaque_wings(nodes):
                # Keep valid samples, including zeros of a fitted response:
                # changing their transmission still changes detected photons.
                planet, star = densities(nodes)
                finite = jnp.all(jnp.isfinite(planet), axis=(-2, -1))
                finite &= jnp.all(jnp.isfinite(star), axis=-1)
                return (throughput == 0) & ~finite

            masked = jax.lax.cond(
                jnp.any(throughput == 0),
                opaque_wings,
                lambda nodes: jnp.zeros_like(nodes, dtype=bool),
                wave,
            )
            # A zero cotangent times an undefined spectral slope can give NaN.
            # Sample opaque, undefined nodes at a transmitted wavelength before
            # differentiation. Their flux and throughput derivative are zero.
            index = jnp.argmax(throughput > 0, axis=1, keepdims=True)
            reference = jnp.take_along_axis(wave, index, axis=1)
            wave = jnp.where(masked, reference, wave)
            throughput = jnp.where(masked, 0.0, throughput)

        planet, star = densities(wave)
        if transmission is not None:
            planet = planet * throughput[..., None, None]
            star = star * throughput[..., None]
        return quadrature.integrate(planet) / quadrature.integrate(star)[:, None, :]

    @_batch_method
    def planet_flux_densities(
        self, wavelength_nm: Array, time_d: Array, *, ephemeris=None
    ) -> Array:
        """Per-planet flux density [ph/s/m^2/nm], shape ``(K_total, T)``.

        ``time_d`` is days from the scene epoch. A point read, like
        :meth:`contrasts`; band quantities come from :meth:`sources`.

        Args:
            wavelength_nm: A single wavelength [nm].
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            Planet photon flux density at the observer [ph s^-1 m^-2 nm^-1],
            shape ``(K_total, T)``.
        """
        return self._photometry(wavelength_nm, time_d, ephemeris, flux=True)

    @_batch_method
    def alpha_dMag(
        self, time_d: Array, *, wavelength_nm: float = 600.0, ephemeris=None
    ) -> tuple[Array, Array]:
        """Per-planet separation + dMag at wavelength_nm (default 600 nm).

        ``time_d`` is days from the scene epoch.

        Each result has shape ``(K_total, T)``; wavelength_nm is scalar.

        Args:
            time_d: Observation times, days from the scene epoch, shape ``(T,)``.
            wavelength_nm: Wavelength at which the contrast is taken [nm].
            ephemeris: Optional ephemeris from :meth:`ephemeris` solved at these
                dates; None solves it here.

        Returns:
            A tuple of the star-planet angular separation [arcsec] and the
            planet-star magnitude difference dMag, each of shape ``(K_total, T)``.
        """
        eph = self.ephemeris(time_d) if ephemeris is None else ephemeris
        alpha = eph.separation_arcsec(eph.scene_planet_indices, 0)
        contrast = self.contrasts(jnp.asarray(wavelength_nm), time_d, ephemeris=eph)
        safe = jnp.where(self.valid[:, None], contrast, 1.0)
        dMag = contrast_to_delta_mag(safe + jnp.finfo(contrast.dtype).tiny)
        return alpha, dMag

    def __repr__(self) -> str:
        """Tree-shaped summary: star + planets + disk + midplane geometry.

        Returns:
            A multi-line text summary.
        """
        if self.batch_shape:
            return (
                f"Scene(batch_shape={self.batch_shape}, capacity={self.capacity}, "
                f"star={type(self.star).__name__}, "
                f"model={type(self.planets[0].physical_model).__name__})"
            )
        n_modules = len(self.planets)
        n_total = self.n_planets
        lines = [
            (
                f"Scene(n_planet_modules={n_modules}, "
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
        if self.local_zodi is not None:
            lines.append(indent("local_zodi: " + repr(self.local_zodi)))
        else:
            lines.append("  local_zodi: None")
        return "\n".join(lines)

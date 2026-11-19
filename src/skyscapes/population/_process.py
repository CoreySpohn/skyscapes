"""Independent stellar occurrence processes and their pre-pruning mark density."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import priorities
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.epoch import Epoch
from orbix.system import Body, SkyFrame
from orbix.system import System as DynamicsSystem
from orbix.system.core import GM_SUN_AU3_PER_D2, enclosed_mu, jacobi_to_barycentric
from orbix.system.frame_orbit import FrameOrbit

from ..physical_model import AbstractPhysicalModel, LambertianPhysicalModel
from ..scene import Planet, Scene
from ..scene.scene import _mask_dynamics
from ..scene.star import AbstractStar
from ._draw import (
    _ANGLE,
    _EARTH_MASSES_PER_SUN,
    _ISOTROPIC_INCLINATION,
    PopulationConfig,
    _draw_tilt,
    _remove_unstable,
    _sample_types,
    _sky_elements_one,
    _type_weights,
    earth_like,
)
from ._tables import OccurrenceTable, PlanetTypes, _type_albedo, mass_to_radius

_MASS_RATIO = Mearth2kg / Msun2kg


class HostStars(eqx.Module):
    """Hosts with stable identity, all leaves carrying one leading axis (S,).

    star_id is int32; mass_Msun, luminosity_Lsun and dist_pc have shape (S,).
    star supplies photometry only. Its numeric leaves, including any spectral
    coordinates, must carry the same leading axis. IDs must be unique and
    stable when selecting or reordering hosts. Mass, luminosity and distance
    must be positive.
    """

    star_id: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.int32))
    mass_Msun: jax.Array = eqx.field(converter=jnp.asarray)
    luminosity_Lsun: jax.Array = eqx.field(converter=jnp.asarray)
    dist_pc: jax.Array = eqx.field(converter=jnp.asarray)
    star: AbstractStar

    def __check_init__(self):
        """Validate static host shapes without inspecting traced values."""
        shape = self.star_id.shape
        if len(shape) != 1 or any(
            x.shape != shape
            for x in (self.mass_Msun, self.luminosity_Lsun, self.dist_pc)
        ):
            raise ValueError("HostStars scalar fields must have shape (S,)")
        if any(jnp.shape(x)[:1] != shape for x in jax.tree.leaves(self.star)):
            raise ValueError("HostStars.star leaves must have leading shape (S,)")


class _TypeReflector(AbstractPhysicalModel):
    """One spectral reflector class, distinguished by a per-planet type leaf."""

    type_index: jax.Array
    wavelengths_nm: jax.Array
    spectrum: jax.Array

    def geometric_albedo(self, wavelength_nm):
        """Interpolate the selected type spectra at a scalar wavelength in nm."""
        return _type_albedo(
            self.type_index, wavelength_nm, self.wavelengths_nm, self.spectrum
        )

    def contrast(
        self, phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
    ):
        """Use the existing Lambertian phase law with the type's spectral albedo."""
        return LambertianPhysicalModel(self.geometric_albedo(wavelength_nm)).contrast(
            phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
        )


def _relative_orbits(system):
    """Recover star-relative epoch states from one masked Jacobi system."""
    dynamics = _mask_dynamics(system.dynamics, system._chain_valid())
    r, v = dynamics.orbits.at_each(jnp.asarray(0.0))

    def relative(x):
        bary = jacobi_to_barycentric(
            x, dynamics.star.mass_Msun, dynamics.planets.mass_Msun
        )
        return (bary[1:] - bary[:1])[jnp.asarray(system.body_index)]

    mass = dynamics.planets.mass_Msun[jnp.asarray(system.body_index)]
    return relative(r), relative(v), mass


class OccurrencePopulation(eqx.Module):
    """Poisson occurrence per host, with a shared sampler and mark density.

    Each cell has an independent Poisson count of mean occurrence.rate;
    cells with bin_id < 0 are excluded, as in the replica. This replaces
    the replica's joint fixed-total rule. Marks are log-uniform in cell
    scaled separation and mass, uniform in orbital angles, with eccentricity drawn
    from config.eccentricity (circular when None),
    with the replica's conditional type probabilities and mass-radius law.

    sample prunes each host alone, without refill, then keeps the innermost
    config.max_planets. The replica truncates before each stability pass
    and refills across hosts. Both remove the outer member on a mass tie
    and stably retain the first member on a separation tie. The pruned
    process has different mark marginals: mark_log_density and log_intensity
    describe the pre-pruning process, not the joint density of sample.

    epoch is static host metadata. Config integers and requested output
    shapes are static; use eqx.filter_jit for methods with integer arguments.
    """

    occurrence: OccurrenceTable
    types: PlanetTypes
    config: PopulationConfig
    epoch: Epoch = eqx.field(static=True)

    def _rates(self):
        return jnp.where(self.occurrence.bin_id >= 0, self.occurrence.rate, 0.0)

    def rate(self, stars):
        """Expected pre-pruning planets per host, shape (S,)."""
        return jnp.broadcast_to(self._rates().sum(), stars.star_id.shape)

    def log_rate(self, stars):
        """Log pre-pruning rate, shape (S,); zero rate gives -inf."""
        return jnp.log(self.rate(stars))

    def _marks(self, key, n, luminosity):
        """Return n independent classical marks and type indices for one host."""
        keys = jax.random.split(key, 8)
        cumulative = jnp.cumsum(self._rates().ravel())
        cumulative = eqx.error_if(
            cumulative,
            cumulative[-1] <= 0,
            "mark_sample requires a positive total rate",
        )
        cell = jnp.minimum(
            jnp.searchsorted(
                cumulative,
                jax.random.uniform(keys[0], (n,)) * cumulative[-1],
                side="right",
            ),
            cumulative.size - 1,
        )
        ia, im = (
            cell // self.occurrence.rate.shape[1],
            cell % self.occurrence.rate.shape[1],
        )
        a_edges, m_edges = self.occurrence.a_edges_AU, self.occurrence.mass_edges_Mearth
        a_scaled = priorities.LogUniform(a_edges[ia], a_edges[ia + 1]).sample(keys[1])
        mass = priorities.LogUniform(m_edges[im], m_edges[im + 1]).sample(keys[2])
        node, peri, anomaly = (
            _ANGLE.sample(k, (n,)) for k in jax.random.split(keys[3], 3)
        )
        ecc = self.config.eccentricity
        e = jnp.zeros(n) if ecc is None else ecc.sample(keys[4], (n,))
        inc = _ISOTROPIC_INCLINATION.sample(keys[5], (n,))
        types = _sample_types(
            keys[6],
            self.types,
            self.config,
            a_scaled,
            mass_to_radius(mass),
            jnp.ones(n, bool),
        )
        return dict(
            a_AU=a_scaled * jnp.sqrt(luminosity),
            mass=mass,
            e=e,
            i_rad=inc,
            W_rad=node,
            w_rad=peri,
            M0_rad=anomaly,
            type_index=types,
        )

    def _system(self, marks, valid, star, mass_msun, dist_pc, mid_inc=0.0, mid_pa=0.0):
        """Put star-relative mark states into the scene's Jacobi hierarchy."""
        mass = jnp.where(valid, marks["mass"] * _MASS_RATIO, 0.0)
        elements = {
            name: marks[name]
            for name in ("a_AU", "e", "i_rad", "W_rad", "w_rad", "M0_rad")
        }
        relative = FrameOrbit.from_classical(
            **elements, mu_AU3_per_d2=GM_SUN_AU3_PER_D2 * (mass_msun + mass)
        )
        r, v = relative.to_cartesian()
        inner_mass = mass_msun + jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)[:-1]])

        def jacobi(x):
            weighted = mass[:, None] * x
            inner = jnp.concatenate(
                [jnp.zeros((1, 3)), jnp.cumsum(weighted, axis=0)[:-1]]
            )
            return x - inner / inner_mass[:, None]

        orbit = FrameOrbit.from_cartesian(
            jacobi(r), jacobi(v), mu_AU3_per_d2=enclosed_mu(mass_msun, mass)
        )
        dynamics = _mask_dynamics(
            DynamicsSystem(
                Body(mass_msun), Body(mass), orbit, SkyFrame(dist_pc, self.epoch)
            ),
            valid,
        )
        model = _TypeReflector(
            jnp.where(valid, marks["type_index"], 0),
            self.types.wavelength_nm,
            self.types.albedo,
        )
        return Scene(
            star,
            Planet(jnp.where(valid, mass_to_radius(marks["mass"]), 0.0), model),
            dynamics,
            valid=valid,
            midplane_inc_deg=mid_inc,
            midplane_position_angle_deg=(90.0 + mid_pa) % 360.0,
        )

    def _sample_star(self, key, star):
        """Draw all candidates, prune, then gather the innermost capacity slots."""
        # The Poisson count is never clipped or redrawn. The static allocation
        # has a checked tail, independent of the final output capacity.
        mean = self.occurrence.total_rate
        capacity = math.ceil(mean + 12 * math.sqrt(mean) + 64)
        count_key, mark_key, plane_key, tilt_key = jax.random.split(key, 4)
        count = jax.random.poisson(count_key, self._rates().sum())
        count = eqx.error_if(
            count,
            count > capacity,
            "Poisson candidate buffer overflow; rebuild with a larger total_rate "
            "allocation bound (the count was not truncated or redrawn)",
        )
        # A zero-rate process is empty; the unused marks still need safe cells.
        mark_rates = jnp.where(
            self._rates().sum() > 0, self._rates(), jnp.ones_like(self.occurrence.rate)
        )
        mark_table = eqx.tree_at(
            lambda t: (t.rate, t.bin_id),
            self.occurrence,
            (mark_rates, jnp.zeros_like(self.occurrence.bin_id)),
        )
        marker = eqx.tree_at(lambda p: p.occurrence, self, mark_table)
        marks = marker._marks(mark_key, capacity, star.luminosity_Lsun)
        order = jnp.argsort(
            jnp.where(jnp.arange(capacity) < count, marks["a_AU"], jnp.inf), stable=True
        )
        marks = jax.tree.map(lambda x: x[order], marks)
        present = jnp.arange(capacity) < count
        alive = _remove_unstable(
            marks["a_AU"],
            marks["mass"],
            present,
            3 * star.mass_Msun * _EARTH_MASSES_PER_SUN,
            self.config.mutual_hill_min,
        )
        selected = jnp.argsort(~alive, stable=True)
        k = self.config.max_planets
        take = selected[jnp.minimum(jnp.arange(k), capacity - 1)]
        valid = jnp.arange(k) < jnp.sum(alive)
        marks = jax.tree.map(lambda x: x[take], marks)
        k_inc, k_pa = jax.random.split(plane_key)
        mid_inc, mid_pa = _ISOTROPIC_INCLINATION.sample(k_inc), _ANGLE.sample(k_pa)
        # Orient surviving marks about a common isotropic system midplane.
        # Tilt is drawn per candidate so selecting K cannot change it.
        tilt = _draw_tilt(self.config, tilt_key, (capacity,))[order][take]
        e, inc, node, peri, anomaly, _ = jax.vmap(
            _sky_elements_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None)
        )(
            marks["a_AU"],
            marks["e"],
            tilt,
            marks["W_rad"],
            marks["w_rad"],
            marks["M0_rad"],
            marks["mass"],
            star.mass_Msun,
            mid_inc,
            mid_pa,
        )
        marks.update(e=e, i_rad=inc, W_rad=node, w_rad=peri, M0_rad=anomaly)
        return self._system(
            marks,
            valid,
            star.star,
            star.mass_Msun,
            star.dist_pc,
            jnp.degrees(mid_inc),
            jnp.degrees(mid_pa),
        )

    def sample(self, key, stars, n_worlds):
        """Draw batch (n_worlds,S) with capacity config.max_planets, inner first.

        The exact host key is fold_in(fold_in(key,w),star_id). Sequential maps
        preserve bits under host selection/reordering and world-count changes.
        n_worlds, S, K and table shapes are static compilation dimensions.
        Poisson storage is ceil(total_rate+12*sqrt(total_rate)+64) candidates;
        overflow raises explicitly, never truncates or redraws. total_rate is
        a host allocation bound and must be rebuilt if the rate grid changes.
        The default rate about 11 gives 115 candidates (tail below 1e-73 per
        host). This finite-storage failure is distinct from keeping the K
        innermost survivors. Zero-rate hosts return fully padded systems.
        """

        def world(w):
            world_key = jax.random.fold_in(key, w)
            return jax.lax.map(
                lambda host: self._sample_star(
                    jax.random.fold_in(world_key, host.star_id), host
                ),
                stars,
            )

        return jax.lax.map(world, jnp.arange(n_worlds))

    def mark_sample(self, key, n, star):
        """Draw n one-planet Scenes, batch (n,), K=1, with no pruning.

        star must be a HostStars of length one. Each draw uses the exact
        isotropic single-planet marginal, including for zero midplane tilt.
        n is a static integer; the rate must be positive.
        """
        if star.star_id.shape != (1,):
            raise ValueError("mark_sample requires HostStars of length one")
        marks = self._marks(key, n, star.luminosity_Lsun[0])
        host = jax.tree.map(lambda x: x[0], star.star)
        return jax.vmap(
            lambda row: self._system(
                jax.tree.map(lambda x: x[None], row),
                jnp.ones(1, bool),
                host,
                star.mass_Msun[0],
                star.dist_pc[0],
            )
        )(marks)

    def mark_read(self, system):
        """Read a capacity-one scene as (relative orbit, mass_Mearth, label).

        Each returned scalar leaf has the scene batch shape B; the orbit
        quaternion has shape B + (4,). The state is at scene time zero.
        Padded entries have finite filler, zero mass and int32 label zero.
        """
        if system.capacity != 1:
            raise ValueError("mark_read requires capacity one")

        def one(scene):
            r, v, mass = _relative_orbits(scene)
            orbit = FrameOrbit.from_cartesian(
                r[0],
                v[0],
                mu_AU3_per_d2=GM_SUN_AU3_PER_D2 * (scene.star_mass_Msun + mass[0]),
            )
            label = scene.planets[0].physical_model.type_index[0]
            return (
                orbit,
                mass[0] / _MASS_RATIO,
                jnp.where(scene.valid[0], label, 0).astype(jnp.int32),
            )

        read = one
        for _ in system.batch_shape:
            read = jax.vmap(read)
        return read(system)

    def mark_build(self, star, orbit, mass_Mearth, label):
        """Build capacity-one scenes with batch B from star-relative marks.

        star is a HostStars of length one; orbit scalar leaves, mass_Mearth
        and int32 label have shape B. The supplied orbit epoch is in days
        from the scene epoch. Stellar and planet masses set the dynamics rate;
        radius and reflection use the same relations as mark_sample.
        """
        if star.star_id.shape != (1,):
            raise ValueError("mark_build requires HostStars of length one")
        mass_Mearth, label = jnp.asarray(mass_Mearth), jnp.asarray(label, jnp.int32)
        if mass_Mearth.shape != orbit.a_AU.shape or label.shape != orbit.a_AU.shape:
            raise ValueError("orbit, mass_Mearth and label must share batch shape B")
        host = jax.tree.map(lambda x: x[0], star.star)

        def one(relative, mass, index):
            marks = dict(relative.to_classical(), mass=mass, type_index=index)
            scene = self._system(
                jax.tree.map(lambda x: x[None], marks),
                jnp.ones(1, bool),
                host,
                star.mass_Msun[0],
                star.dist_pc[0],
            )
            # For one planet the relative and Jacobi states coincide. Keeping
            # the supplied frame avoids an unnecessary inverse-angle round trip.
            mu = GM_SUN_AU3_PER_D2 * (star.mass_Msun[0] + mass * _MASS_RATIO)
            relative = eqx.tree_at(
                lambda o: o.n_rad_per_d, relative, jnp.sqrt(mu / relative.a_AU**3)
            )
            return eqx.tree_at(
                lambda s: s.dynamics.orbits,
                scene,
                jax.tree.map(lambda x: x[None], relative),
            )

        build = one
        for _ in orbit.a_AU.shape:
            build = jax.vmap(build)
        return build(orbit, mass_Mearth, label)

    def _density_one(self, system, luminosity):
        r, v, mass_sun = _relative_orbits(system)
        mu = GM_SUN_AU3_PER_D2 * (system.star_mass_Msun + mass_sun)
        distance = jnp.linalg.norm(r, axis=-1)
        energy = 2 / distance - jnp.sum(v * v, axis=-1) / mu
        a = 1 / jnp.where(energy > 0, energy, 1.0)
        evec = jnp.cross(v, jnp.cross(r, v)) / mu[:, None] - r / distance[:, None]
        evec = jnp.where(system.valid[:, None], evec, jnp.array([0.2, 0.0, 0.0]))
        eccentricity = jnp.linalg.norm(evec, axis=-1)
        m = mass_sun / _MASS_RATIO
        safe_m = jnp.where(m > 0, m, 1.0)
        scaled = a / jnp.sqrt(luminosity)
        table = self.occurrence
        ia = jnp.searchsorted(table.a_edges_AU, scaled, side="right") - 1
        im = jnp.searchsorted(table.mass_edges_Mearth, safe_m, side="right") - 1
        in_grid = (
            (ia >= 0)
            & (ia < table.rate.shape[0])
            & (im >= 0)
            & (im < table.rate.shape[1])
        )
        ia = jnp.clip(ia, 0, table.rate.shape[0] - 1)
        im = jnp.clip(im, 0, table.rate.shape[1] - 1)
        rates = self._rates()
        cell = rates[ia, im]
        widths = jnp.log(table.a_edges_AU[ia + 1] / table.a_edges_AU[ia]) * jnp.log(
            table.mass_edges_Mearth[im + 1] / table.mass_edges_Mearth[im]
        )
        weights = _type_weights(self.types, self.config, scaled, mass_to_radius(safe_m))
        total = weights.sum(axis=-1)
        index = system.planets[0].physical_model.type_index
        fallback = self.types.names.index("Mars") if "Mars" in self.types.names else 0
        prob = jnp.take_along_axis(
            weights,
            jnp.clip(index, 0, len(self.types.names) - 1).astype(jnp.int32)[:, None],
            -1,
        )[:, 0]
        prob = jnp.where(
            total > 0, prob / jnp.where(total > 0, total, 1.0), index == fallback
        )
        ecc_logp = self.config.eccentricity.log_prob(eccentricity)
        support = (
            in_grid
            & (cell > 0)
            & (m > 0)
            & (energy > 0)
            & jnp.isfinite(ecc_logp)
            & (index == index.astype(jnp.int32))
            & (index >= 0)
            & (index < len(self.types.names))
            & (prob > 0)
        )
        safe_e = jnp.where(eccentricity > 0, eccentricity, 1.0)
        logp = (
            jnp.log(jnp.where(cell > 0, cell, 1.0))
            - jnp.log(rates.sum())
            - jnp.log(widths)
            - jnp.log(safe_m)
            - jnp.log(safe_e)
            + jnp.where(jnp.isfinite(ecc_logp), ecc_logp, 0.0)
            - jnp.log(2 * (2 * jnp.pi) ** 3)
            + jnp.log(jnp.where(prob > 0, prob, 1.0))
        )
        return jnp.where(system.valid, jnp.where(support, logp, -jnp.inf), 0.0)

    def mark_log_density(self, system, stars):
        """Log mark density, B + (K,), against dnu dMearth and type counting.

        dnu is the reference measure of ClassicalChart.log_volume with its
        documented omitted constant: unnormalized Haar times dex dey dln(a).
        Write da=ln(a_hi/a_lo), dm=ln(m_hi/m_lo), de=1/p_e with p_e the
        eccentricity density, p_t=type prob.
        q(ln a,e,c,W,w,M,m,t) = (rate_cell/rate)*p_t/[da*dm*m*de*2*(2*pi)^3].
        For x=(ln a,logit e,atanh c,W,w,M), q_x=q*e*(1-e)*(1-c*c).
        exp(ClassicalChart.log_volume(x))=e*e*(1-e)*(1-c*c).
        Thus log p_nu=log q_x-log_volume=log q-log e, evaluated on the
        star-relative epoch orbit with mu=G*(system star mass + planet mass).
        For normalized Haar instead, add log(8*pi*pi) to this log density.

        An isotropic midplane followed by an azimuthally uniform tilt has
        a rotation-invariant normal distribution; the unique invariant law
        on the sphere is uniform. Uniform periapsis and anomaly supply the
        remaining independent angles. This proof includes any fixed tilt,
        so any inclination distribution leaves this marginal nonsingular.
        Circular orbits (config.eccentricity None) make it singular and raise
        ValueError on the host, also at trace construction.

        stars has length S matching the last batch axis, or length one to
        broadcast a single host across B. Padded slots return zero. Radius
        is the deterministic mass-radius image, not an independent coordinate.
        """
        if self.config.eccentricity is None:
            raise ValueError(
                "config.eccentricity is None (circular orbits), which has no "
                "full-dimensional density; set an eccentricity distribution"
            )
        luminosity = (
            jnp.broadcast_to(stars.luminosity_Lsun, system.batch_shape)
            if system.batch_shape
            else stars.luminosity_Lsun[0]
        )
        read = self._density_one
        for _ in system.batch_shape:
            read = jax.vmap(read)
        return read(system, luminosity)

    def log_intensity(self, system, stars):
        """Log pre-pruning intensity, B + (K,); padding remains exactly zero."""
        log_rate = (
            jnp.broadcast_to(self.log_rate(stars), system.batch_shape)
            if system.batch_shape
            else self.log_rate(stars)[0]
        )
        value = self.mark_log_density(system, stars) + log_rate[..., None]
        return jnp.where(system.valid, value, 0.0)

    def earth_like(self, system, stars):
        """Whether each planet is an exoEarth candidate, bool B + (K,).

        Applies :func:`earth_like` to the star-relative osculating semi-major
        axis at scene time zero, the mass-radius image of the planet mass and
        the host luminosity. stars has length S matching the last batch axis,
        or length one to broadcast. Padded and unbound slots return False.
        """
        luminosity = (
            jnp.broadcast_to(stars.luminosity_Lsun, system.batch_shape)
            if system.batch_shape
            else stars.luminosity_Lsun[0]
        )

        def one(scene, lum):
            r, v, mass_sun = _relative_orbits(scene)
            mu = GM_SUN_AU3_PER_D2 * (scene.star_mass_Msun + mass_sun)
            energy = 2 / jnp.linalg.norm(r, axis=-1) - jnp.sum(v * v, axis=-1) / mu
            bound = scene.valid & (energy > 0) & (mass_sun > 0)
            a = 1 / jnp.where(bound, energy, 1.0)
            radius = mass_to_radius(jnp.where(bound, mass_sun / _MASS_RATIO, 1.0))
            return bound & earth_like(a, radius, lum)

        read = one
        for _ in system.batch_shape:
            read = jax.vmap(read)
        return read(system, luminosity)

    def type_albedo(self, system, wavelength_nm):
        """Geometric albedo, B + (K,), at scalar wavelength_nm; padding is zero.

        The scene's physical model also carries these spectra and the type
        index, so ordinary scene photometry uses the same albedo automatically.
        """
        index = system.planets[0].physical_model.type_index
        return jnp.where(
            system.valid, self.types.geometric_albedo(index, wavelength_nm), 0.0
        )

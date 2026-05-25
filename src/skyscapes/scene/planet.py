"""scene.Planet -- composes an AbstractOrbit with an AbstractPhysicalModel.

Planet-intrinsic parameters (radius, mass) live on ``Planet`` itself.
Submodels receive them as method arguments when needed:

  - ``orbit`` owns orbital elements (a, e, i, ...). Those parameters
    are the orbit's parameterization, not the planet's identity.
  - ``physical_model`` owns spectral physics (albedo, contrast grid,
    ExoJax composition). Planet radius is passed to
    ``physical_model.contrast(..., Rp_Rearth=self.Rp_Rearth)`` rather
    than duplicated as a field.

Stellar context (``Ms_kg``, ``dist_pc``) is supplied keyword-only at
call time through a ``Star`` argument -- ``Planet`` never stores a
reference to its host star, which keeps the PyTree shallow and lets a
single ``scene.System`` own the one-and-only star.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import G, pc2AU, rad2arcsec, two_pi
from jaxtyping import Array
from orbix.equations.orbit import mean_anomaly_tp, mean_motion, period_n
from orbix.system.orbit import AbstractOrbit

from .._repr import fmt_scalar_or_array, indent
from ..physical_model import AbstractPhysicalModel
from .star import AbstractStar


class Planet(eqx.Module):
    """Composed planet: intrinsic params + orbit dynamics + physical-model physics.

    All stellar-context-dependent methods take a ``star`` keyword argument
    rather than holding a reference internally.  This keeps ``System`` as
    the single source of truth for the host star.

    Attributes:
        Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
        Mp_Mearth: Planet mass [Earth masses], shape ``(K,)``.
        orbit: Orbital dynamics (trajectory parameterization).
        physical_model: Spectral physics (reflectivity / emission).
    """

    Rp_Rearth: Array
    Mp_Mearth: Array
    orbit: AbstractOrbit
    physical_model: AbstractPhysicalModel

    @property
    def n_planets(self) -> int:
        """Number of planets ``K`` carried by this composed module."""
        return int(self.Rp_Rearth.shape[0])

    def mean_anomaly(self, t_jd: Array, *, star: AbstractStar) -> Array:
        """Mean anomaly mod 360 [deg], shape ``(K, T)``.

        ``t_jd`` must be shape ``(T,)`` -- no rank polymorphism. Callers
        that hold a scalar should wrap it in ``jnp.asarray([t])`` at the
        call site.
        """
        n = mean_motion(self.orbit.a_AU, G * star.Ms_kg)
        T_d = period_n(n)
        tp_d = self.orbit.t0_d - T_d * self.orbit.M0_rad / two_pi
        M = mean_anomaly_tp(t_jd[None, :], n[:, None], tp_d[:, None]) % two_pi
        return jnp.rad2deg(M)

    def propagate(self, trig_solver, t_jd: Array, *, star: AbstractStar):
        """Delegate to ``orbit.propagate``; returns ``(r_AU, phase_rad, dist_AU)``."""
        return self.orbit.propagate(trig_solver, t_jd, Ms_kg=star.Ms_kg)

    def position_arcsec(
        self,
        trig_solver,
        t_jd: Array,
        *,
        star: AbstractStar,
    ) -> Array:
        """On-sky position, shape ``(2, K, T)`` -- (dRA, dDec) in arcsec."""
        r_AU, _, _ = self.propagate(trig_solver, t_jd, star=star)
        dist_AU = star.dist_pc * pc2AU
        scale = rad2arcsec / dist_AU
        ra = r_AU[:, 0, :] * scale
        dec = r_AU[:, 1, :] * scale
        return jnp.stack([ra, dec], axis=0)

    def alpha_dMag(
        self,
        trig_solver,
        t_jd: Array,
        *,
        star: AbstractStar,
        wavelength_nm: float = 600.0,
    ) -> tuple[Array, Array]:
        """Projected separation [arcsec] and delta-mag, each ``(K, T)``.

        For ``LambertianPhysicalModel`` (grey), the chosen ``wavelength_nm``
        is irrelevant and the output matches ``orbix.Planets.alpha_dMag``
        by construction. For ``GridPhysicalModel`` / future wavelength-
        dependent models, ``dMag`` is evaluated at ``wavelength_nm``;
        pick a value within the model's spectral grid.
        """
        r_AU, phase_angle_rad, dist_AU = self.propagate(trig_solver, t_jd, star=star)
        s_AU = jnp.sqrt(r_AU[:, 0, :] ** 2 + r_AU[:, 1, :] ** 2)
        dist_pc_AU = star.dist_pc * pc2AU
        alpha = s_AU * (rad2arcsec / dist_pc_AU)

        contrast = self.physical_model.contrast(
            phase_angle_rad=phase_angle_rad,
            dist_AU=dist_AU,
            wavelength_nm=jnp.asarray(wavelength_nm),
            Rp_Rearth=self.Rp_Rearth,
        )
        eps = jnp.finfo(contrast.dtype).tiny
        dMag = -2.5 * jnp.log10(contrast + eps)
        return alpha, dMag

    def contrast(
        self,
        trig_solver,
        wavelength_nm: Array,
        t_jd: Array,
        *,
        star: AbstractStar,
    ) -> Array:
        """Planet-to-star contrast at (wavelength, time), shape ``(K, T)``."""
        _, phase_angle_rad, dist_AU = self.propagate(trig_solver, t_jd, star=star)
        return self.physical_model.contrast(
            phase_angle_rad=phase_angle_rad,
            dist_AU=dist_AU,
            wavelength_nm=wavelength_nm,
            Rp_Rearth=self.Rp_Rearth,
        )

    def spec_flux_density(
        self,
        trig_solver,
        wavelength_nm: Array,
        t_jd: Array,
        *,
        star: AbstractStar,
    ) -> Array:
        """Planet flux density [ph/s/m^2/nm], shape ``(K, T)``."""
        c = self.contrast(trig_solver, wavelength_nm, t_jd, star=star)
        f_star = star.spec_flux_density(wavelength_nm, t_jd)
        return c * f_star

    def __repr__(self) -> str:
        """Nested summary: planet count + intrinsic params + orbit + physical_model."""
        rp = fmt_scalar_or_array(self.Rp_Rearth)
        mp = fmt_scalar_or_array(self.Mp_Mearth)
        return (
            f"Planet(K={self.n_planets}, Rp={rp} R_earth, Mp={mp} M_earth)\n"
            f"{indent('orbit: ' + repr(self.orbit))}\n"
            f"{indent('physical_model: ' + repr(self.physical_model))}"
        )

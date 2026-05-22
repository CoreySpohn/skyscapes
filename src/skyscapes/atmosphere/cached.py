"""Pre-computed reflectivity atmosphere.

Holds a fixed ``(K, n_nu)`` plane-parallel-reflectivity array that was
produced by running some other :class:`AbstractAtmosphere`'s heavy
radiative transfer once. ``reflected_spectrum`` becomes a cubic
interpolation + Lambert phase + ``(Rp/d)^2`` lookup -- microseconds
per call vs. seconds for an ExoJAX 2-stream RT.

Use cases:
  - Coronagraphoto simulations where the atmosphere is fixed across
    many evaluations (orbits, wavelengths, time samples).
  - ETC studies where you want fast forward-modeling against a
    canonical Earth-like atmosphere.

Not for HMC retrievals where atmosphere parameters vary -- the
underlying expensive atmosphere is the right tool there.

The cache file format is a NumPy ``.npz`` with three keyword arrays
(``Rp_Rearth``, ``reflectivity``, ``nu_grid``) and a small
JSON-style header tag for version-tracking.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import (
    nm_to_wavenumber_cm,
    spherical_to_geometric_albedo,
)
from jaxtyping import Array

from .._repr import indent
from .base import AbstractAtmosphere
from .lambertian import _lambert_phase

# Bump when the cache file layout changes in a way that invalidates
# previously-saved caches.
CACHE_FORMAT_VERSION = 1


class PrecomputedReflectivity(AbstractAtmosphere):
    """Atmosphere whose reflectivity spectrum has been pre-computed.

    Attributes:
        Rp_Rearth: Planetary radii [Earth radii], shape ``(K,)``.
        reflectivity: Per-planet plane-parallel reflectivity ``R(nu)``
            on the wavenumber grid, shape ``(K, n_nu)``. Already
            includes all the physics of the original atmosphere
            (absorption, Rayleigh, clouds, surface).
        nu_grid: Wavenumber grid [cm^-1], shape ``(n_nu,)``.
        n_nu: Length of ``nu_grid`` (static for JIT).
    """

    Rp_Rearth: Array
    reflectivity: Array
    nu_grid: Array
    n_nu: int = eqx.field(static=True)

    @classmethod
    def from_atmosphere(cls, atm: AbstractAtmosphere) -> PrecomputedReflectivity:
        """Pre-compute the reflectivity from an existing atmosphere.

        Calls the atmosphere's internal ``_reflectivity_all_planets``
        once and packages the result. Requires the atmosphere to
        expose ``_reflectivity_all_planets``, ``Rp_Rearth``,
        ``nu_grid``, and ``n_nu`` -- presently :class:`ExoJaxAtmosphere`
        is the supported source.
        """
        try:
            reflectivity = atm._reflectivity_all_planets()
        except AttributeError as e:  # pragma: no cover
            raise TypeError(
                f"{type(atm).__name__} does not expose "
                "_reflectivity_all_planets; PrecomputedReflectivity only "
                "supports atmospheres that compute a per-planet R(nu) "
                "spectrum (currently ExoJaxAtmosphere)."
            ) from e
        return cls(
            Rp_Rearth=atm.Rp_Rearth,
            reflectivity=reflectivity,
            nu_grid=atm.nu_grid,
            n_nu=atm.n_nu,
        )

    @classmethod
    def load(cls, path: str | Path) -> PrecomputedReflectivity:
        """Load a previously-saved cache file.

        Args:
            path: Path to a ``.npz`` file produced by :meth:`save`.

        Returns:
            A ``PrecomputedReflectivity`` ready to evaluate.

        Raises:
            ValueError: if the file's ``cache_format_version`` differs
                from the current code's version (would silently
                produce wrong spectra otherwise).
        """
        path = Path(path)
        with np.load(path) as data:
            version = int(data["cache_format_version"])
            if version != CACHE_FORMAT_VERSION:
                raise ValueError(
                    f"Cache file {path} has format version {version}; "
                    f"current code version is {CACHE_FORMAT_VERSION}. "
                    f"Delete the file to rebuild."
                )
            Rp_Rearth = jnp.asarray(data["Rp_Rearth"])
            reflectivity = jnp.asarray(data["reflectivity"])
            nu_grid = jnp.asarray(data["nu_grid"])
        return cls(
            Rp_Rearth=Rp_Rearth,
            reflectivity=reflectivity,
            nu_grid=nu_grid,
            n_nu=int(nu_grid.shape[0]),
        )

    def save(self, path: str | Path) -> None:
        """Save the cached reflectivity to a ``.npz`` file.

        Idempotent and safe to call from JAX-traced contexts (the
        save itself is plain NumPy -- callers should not call this
        inside a JIT region, but the data is just regular arrays).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            cache_format_version=np.asarray(CACHE_FORMAT_VERSION),
            Rp_Rearth=np.asarray(self.Rp_Rearth),
            reflectivity=np.asarray(self.reflectivity),
            nu_grid=np.asarray(self.nu_grid),
        )

    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Per-planet, per-time geometric-albedo contrast at one wavelength.

        Args:
            phase_angle_rad: Star-planet-observer phase angle, shape ``(K, T)``.
            dist_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm].

        Returns:
            Contrast = ``A_g(lambda) * Lambert_phase(beta) * (Rp/d)^2``,
            shape ``(K, T)``. The cached array stores the underlying
            atmosphere's plane-parallel (spherical) reflectivity; we
            convert to geometric albedo via the Lambertian-sphere
            factor 2/3 (Seager 2010, eq 3.36) at call time -- same
            convention as :class:`ExoJaxAtmosphere.reflected_spectrum`.
        """
        target_nu = nm_to_wavenumber_cm(wavelength_nm)

        def interp_one(spectrum):
            return interpax.interp1d(
                target_nu,
                self.nu_grid,
                spectrum,
                method="cubic",
                extrap=True,
            )

        R_at_wl = jax.vmap(interp_one)(self.reflectivity)  # (K,)
        Rp_AU = (self.Rp_Rearth * Rearth2AU)[:, None]
        Ag = spherical_to_geometric_albedo(R_at_wl)[:, None]
        phase = _lambert_phase(phase_angle_rad)
        return Ag * phase * (Rp_AU / dist_AU) ** 2

    def reflected_spectrum_cube(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelengths_nm: Array,
    ) -> Array:
        """Per-planet, per-time geometric-albedo contrast across wavelengths.

        Returns shape ``(W, K, T)``. Avoids per-wavelength
        recomputation by vectorising the interpolation; applies the
        Lambertian-sphere spherical-to-geometric conversion the same
        way as :meth:`reflected_spectrum`.
        """
        target_nu = nm_to_wavenumber_cm(wavelengths_nm)

        def interp_one_planet(spectrum):
            return interpax.interp1d(
                target_nu,
                self.nu_grid,
                spectrum,
                method="cubic",
                extrap=True,
            )

        R_at_wls = spherical_to_geometric_albedo(
            jax.vmap(interp_one_planet)(self.reflectivity)
        )  # (K, W)
        Rp_AU = (self.Rp_Rearth * Rearth2AU)[:, None]
        phase = _lambert_phase(phase_angle_rad)
        geom = phase * (Rp_AU / dist_AU) ** 2  # (K, T)
        return jnp.einsum("kw,kt->wkt", R_at_wls, geom)

    def __repr__(self) -> str:
        """Compact summary of the cached spectrum."""
        K = int(self.Rp_Rearth.shape[0])
        wl_min = float(1.0e7 / self.nu_grid.max())
        wl_max = float(1.0e7 / self.nu_grid.min())
        peak_mean = float(jnp.mean(self.reflectivity.max(axis=-1)))
        lines = [
            f"PrecomputedReflectivity(K={K}, n_nu={self.n_nu})",
            (
                f"  Wavelength: {wl_min:.0f}-{wl_max:.0f} nm, "
                f"mean peak R = {peak_mean:.3f}"
            ),
            indent(
                "(no RT cost; reflectivity sampled at call time)",
                prefix="  ",
            ),
        ]
        return "\n".join(lines)

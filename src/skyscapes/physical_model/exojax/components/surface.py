"""Surface reflectivity components."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .base import AbstractSurface


class WavelengthDependentSurface(AbstractSurface):
    """Wavelength-dependent surface reflectivity.

    The bottom-of-atmosphere reflectivity is
    ``10**log_albedo * spectrum``: the ``log_albedo`` PyTree leaf is a
    per-planet fittable scaling, and ``spectrum`` is a fixed
    wavelength-dependent profile (vegetation red-edge, water
    absorption, snow/ice, ...) shared across planets.

    Attributes:
        log_albedo: Log10 surface albedo scaling per planet, shape ``(K,)``.
        spectrum: Wavelength-dependent reflectivity profile,
            shape ``(n_nu,)``. Defaults to flat ones for "no spectral
            shape; albedo is constant across the band".
    """

    log_albedo: Array
    spectrum: Array

    def compute_refl(self, log_albedo_scalar: Array, n_nu: int) -> Array:
        """Return wavelength-dependent surface reflectivity.

        ``n_nu`` is accepted for parity with :class:`FlatSurface` but
        unused -- the wavelength axis comes from ``self.spectrum``.
        """
        return (10.0**log_albedo_scalar) * self.spectrum


class FlatSurface(AbstractSurface):
    """Wavelength-independent (gray) Lambertian surface.

    Equivalent to ``WavelengthDependentSurface`` with a flat spectrum
    but structurally explicit; useful when callers want to document
    "I am intentionally using a featureless surface".

    Attributes:
        log_albedo: Log10 surface albedo per planet, shape ``(K,)``.
    """

    log_albedo: Array

    def compute_refl(self, log_albedo_scalar: Array, n_nu: int) -> Array:
        """Return scalar albedo broadcast across the wavenumber grid."""
        return (10.0**log_albedo_scalar) * jnp.ones(n_nu)


class EmittingSurface(eqx.Module):
    """Opaque thermal-emitting ground: ``emissivity * B_nu(T_surface)``.

    The ground is a gray or spectrally shaped blackbody with emissivity
    ``1 - reflectance`` (Kirchhoff). Downwelling atmospheric radiation
    reflected by a ground with emissivity below one is neglected, so for
    ``emissivity < 1`` the emergent flux is a lower bound.

    Attributes:
        temperature_K: Surface temperature [K], shape ``(K,)``.
        emissivity: Scalar emissivity per planet, shape ``(K,)``.
        spectrum: Wavelength-dependent emissivity shape on the wavenumber
            grid, shape ``(n_nu,)``; flat ones for a gray ground. The
            emissivity used is ``emissivity[k] * spectrum``.
    """

    temperature_K: Array
    emissivity: Array
    spectrum: Array

    def compute_emissivity(self, k: int) -> Array:
        """Emissivity of planet ``k`` on the wavenumber grid, shape ``(n_nu,)``."""
        return self.emissivity[k] * self.spectrum

"""Abstract physical-model interface for a planet's radiative output.

Subclasses own whatever spectral-physics parameters describe "this kind of
planet" (Lambertian, grid-interpolated, radiative-transfer, future thermal
emission). The only invariant is a single hook that maps
(phase_angle, star-planet distance, wavelength, planet radius) onto a
contrast (flux ratio relative to the host star).

The planet radius is supplied as a method argument rather than a class
field -- ``Rp_Rearth`` is a property of the planet, not the physical
model. ``Planet.contrast`` forwards ``self.Rp_Rearth`` to the physical
model at call time.
"""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array


class AbstractPhysicalModel(eqx.Module):
    """Spectral-physics layer for a planet.

    Concrete subclasses own per-planet physical-model parameters (albedo,
    grid contrast cube, ExoJax composition, etc.). The host planet's
    radius is supplied at call time via ``contrast(..., Rp_Rearth=...)``.
    """

    @abstractmethod
    def contrast(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Return planet-to-star contrast.

        Args:
            phase_angle_rad: Star-planet-observer phase angle ``beta``,
                shape ``(K, T)`` (matching ``AbstractOrbit.propagate``).
            dist_AU: Star-planet distance, shape ``(K, T)``.
            wavelength_nm: Wavelength, scalar or shape ``(W,)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``,
                supplied by the host ``Planet``.

        Returns:
            Flux-ratio contrast, shape ``(K, T)``.
        """

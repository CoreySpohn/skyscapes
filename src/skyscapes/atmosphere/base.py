"""Abstract atmosphere interface.

Subclasses own whatever spectral-physics parameters describe "this kind of
reflected-light source" (Lambertian, grid-interpolated, radiative-transfer).
The only invariant is a planetary radius and a single hook that maps
(phase_angle, star-planet distance, wavelength) onto a contrast (flux ratio
relative to the host star).
"""

from abc import abstractmethod

import equinox as eqx
from equinox import AbstractVar
from jaxtyping import Array


class AbstractAtmosphere(eqx.Module):
    """Spectral-physics layer for a planet.

    Attributes:
        Rp_Rearth: Planetary radius in Earth radii, shape ``(K,)``.
    """

    Rp_Rearth: AbstractVar[Array]

    @abstractmethod
    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Return planet-to-star contrast.

        Args:
            phase_angle_rad: Star-planet-observer phase angle ``beta``,
                shape ``(K,)`` or broadcastable.
            dist_AU: Star-planet distance, shape ``(K,)`` or broadcastable.
            wavelength_nm: Wavelength, scalar or shape ``(W,)``.

        Returns:
            Flux-ratio contrast, shape broadcast of the above.
        """

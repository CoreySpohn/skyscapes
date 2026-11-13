"""Abstract physical-model interface for a planet's radiative output.

Subclasses own whatever spectral-physics parameters describe "this kind of
planet" (Lambertian, grid-interpolated, radiative-transfer, future thermal
emission). The phase-dependent hook maps
(phase_angle, star-planet distance, wavelength, planet radius) onto a
contrast (flux ratio relative to the host star).
Time-dependent models override ``contrast_at_time`` instead of identifying
physical epochs with orbital phase.

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
            wavelength_nm: Scalar wavelength [nm]. Use ``System.contrast_cube``
                to evaluate an array of wavelengths.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``,
                supplied by the host ``Planet``.

        Returns:
            Flux-ratio contrast, shape ``(K, T)``.
        """

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_jd: Array,
        star=None,
    ) -> Array:
        """Return contrast with the observation epochs and the host star available.

        The default forwards the four radiative arguments to ``contrast``
        and ignores ``time_jd`` and ``star``. Existing phase-dependent models
        therefore keep their behavior without implementing another method.
        A model whose planet flux does not scale with the star's (thermal
        emission) overrides this method and divides by the star's flux.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``.
            dist_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm], as accepted by ``contrast``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_jd: Observation Julian Dates, shape ``(T,)``.
            star: The host ``AbstractStar``; ``Planet`` always supplies it.

        Returns:
            Planet-to-star flux ratio, shape ``(K, T)``.
        """
        return self.contrast(phase_angle_rad, dist_AU, wavelength_nm, Rp_Rearth)

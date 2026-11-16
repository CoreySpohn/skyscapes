"""Abstract physical-model interface for a planet's radiative output.

Subclasses own whatever spectral-physics parameters describe "this kind of
planet" (Lambertian, grid-interpolated, radiative-transfer, thermal
emission). ``spec_flux_density`` returns photon flux at the observer.
Its default multiplies contrast by the host star's photon flux; thermal
models provide flux directly. The phase-dependent hook maps
(phase_angle, star-planet distance, wavelength, planet radius) onto a
contrast (flux ratio relative to the host star).
Time-dependent models override ``contrast_at_time`` instead of identifying
physical epochs with orbital phase.

The planet radius is supplied as a method argument rather than a class
field -- ``Rp_Rearth`` is a property of the planet, not the physical
model. ``System`` forwards ``planet.Rp_Rearth`` to the physical
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

    def spec_flux_density(
        self,
        wavelength_nm: Array,
        *,
        phase_angle_rad: Array,
        dist_AU: Array,
        Rp_Rearth: Array,
        time_d: Array,
        star,
        dist_pc: Array,
    ) -> Array:
        """Planet photon flux density at the observer [ph/s/m^2/nm].

        The default multiplies ``contrast_at_time`` by stellar flux.
        Thermal models override this method to evaluate flux directly.

        Args:
            wavelength_nm: Scalar wavelength [nm].
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``.
            dist_AU: Star-planet distance [AU], shape ``(K, T)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            dist_pc: Observer distance supplied by the system [pc].
            star: Host ``AbstractStar``, providing observer flux.

        Returns:
            Photon flux density, shape ``(K, T)``.
        """
        contrast = self.contrast_at_time(
            phase_angle_rad=phase_angle_rad,
            dist_AU=dist_AU,
            wavelength_nm=wavelength_nm,
            Rp_Rearth=Rp_Rearth,
            time_d=time_d,
            star=star,
            dist_pc=dist_pc,
        )
        return contrast * star.spec_flux_density(wavelength_nm, time_d, dist_pc=dist_pc)

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
                shape ``(K, T)`` for K planets and T times.
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
        time_d: Array,
        star=None,
        dist_pc: Array | None = None,
    ) -> Array:
        """Return contrast with the observation epochs and the host star available.

        The default forwards the four radiative arguments to ``contrast``
        and ignores ``time_d`` and ``star``. Existing phase-dependent models
        therefore keep their behavior without implementing another method.
        A model whose planet flux does not scale with the star's (thermal
        emission) overrides this method and divides by the star's flux.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``.
            dist_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm], as accepted by ``contrast``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            dist_pc: Observer distance supplied by the system [pc].
            star: The host ``AbstractStar``; ``System`` always supplies it.

        Returns:
            Planet-to-star flux ratio, shape ``(K, T)``.
        """
        return self.contrast(phase_angle_rad, dist_AU, wavelength_nm, Rp_Rearth)

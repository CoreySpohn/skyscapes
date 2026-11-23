"""Local (solar-system) zodiacal photon radiance, with physics delegated to zodi."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import zodi
from hwoutils.conversions import um_to_nm
from jax import Array
from orbix.observatory import ecliptic_pointing
from orbix.system import SkyFrame
from zodi.tables import TABLE19_WAVELENGTH_UM
from zodi.units import intensity_from_ab_mag_arcsec2, power_to_photon_intensity

from .._flux_scale import check_flux_scale
from .._tables import table_digest


class AbstractLocalZodi(eqx.Module):
    """Local background radiance in ph s^-1 m^-2 nm^-1 sr^-1.

    The model is given where and when the scene is seen, the scene's
    :class:`~orbix.system.SkyFrame` and the time in days from its epoch, and
    derives whatever it needs from them. Every implementation preserves the
    wavelength array's shape.

    ``flux_scale`` is a dimensionless factor, default one, on the local zodi
    alone: it does not scale the star, the planets or the disk. The instrument's
    own ``flux_scale`` also multiplies this light.

    Attributes:
        flux_scale: Dimensionless factor on the local zodi alone, default one.
            It is checked finite, not positive.
    """

    @abstractmethod
    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the model's spectral breakpoints in nm as a host array.

        Returns:
            Breakpoint wavelengths [nm], a NumPy array on the host.
        """

    flux_scale: eqx.AbstractVar[Array]

    def __check_init__(self):
        """Require a finite ``flux_scale``."""
        check_flux_scale(self.flux_scale)

    def photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance at wavelengths in nm, times ``flux_scale``.

        No integration is done. ``flux_scale`` is applied here, once; a concrete
        model implements :meth:`_photon_radiance` and never sees it.

        Args:
            wavelength_nm: Wavelengths [nm].
            sky_frame: The scene's sky frame: position, distance and epoch. A
                model that depends on the sightline raises a ``ValueError``
                when the frame has no position (an unplaced scene).
            time_d: Observation time, days from ``sky_frame.epoch``.

        Returns:
            Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], with the shape of
            ``wavelength_nm``.
        """
        return self.flux_scale * self._photon_radiance(wavelength_nm, sky_frame, time_d)

    @abstractmethod
    def _photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Unscaled photon radiance; the hook a concrete model implements."""

    def node_radiance(
        self, quadrature, resolution, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Node-effective photon radiance on a band rule's nodes, with ``flux_scale``.

        The default samples the nodes. That is exact for a model whose knots
        are piece edges of the rule, as the Table 19 knots of ``AYOZodi`` and
        ``LeinertZodi`` are when the model is declared to
        ``Quadrature.build``.

        Args:
            quadrature: Band rule from ``optixstuff.spectral.Quadrature.build``.
            resolution: ``optixstuff.spectral.Resolution`` built for the rule;
                only a tabulated model reads it.
            sky_frame: The scene's sky frame.
            time_d: Observation time, scalar days from ``sky_frame.epoch``.

        Returns:
            Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape ``(W, Q)``.
        """
        return self.photon_radiance(quadrature.nodes_nm, sky_frame, time_d)


@final
class AYOZodi(AbstractLocalZodi):
    """Fixed 22 AB mag/arcsec^2 at 550 nm with Table 19 spectral color.

    The sky frame and time are accepted and ignored. The reference magnitude
    parameter sets the fixed apparent brightness; zodi owns the color model.

    Attributes:
        surface_brightness_mag_arcsec2: Apparent surface brightness at the
            reference wavelength [AB mag arcsec^-2].
        reference_wavelength_nm: Wavelength at which the brightness applies [nm].
        flux_scale: Dimensionless factor on the local zodi alone, default one.
    """

    surface_brightness_mag_arcsec2: float = 22.0
    reference_wavelength_nm: float = 550.0
    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them.

        Returns:
            Breakpoint wavelengths [nm], a NumPy array on the host.
        """
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def _photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr; the sightline is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        mag = zodi.ayo_default_ab_mag(
            wave,
            mag_v=self.surface_brightness_mag_arcsec2,
            reference_wavelength_nm=self.reference_wavelength_nm,
        )
        return (
            power_to_photon_intensity(intensity_from_ab_mag_arcsec2(mag, wave), wave)
            / 1000.0
        )


@final
class LeinertZodi(AbstractLocalZodi):
    """Sightline-dependent Leinert radiance anchored on absolute Table 19.

    The ecliptic latitude and the helioecliptic longitude (target minus Sun)
    follow from the frame's position and the time through
    :func:`orbix.observatory.ecliptic_pointing`. The absolute scale comes from
    ``zodi.specific_intensity`` with ``anchor='table19'``; there is no
    adjustable reference magnitude. At (longitude=90, latitude=0) the V-band
    brightness is approximately 22 AB mag/arcsec^2; AYO's rounded 22-mag
    normalization differs slightly from the absolute table value.

    Attributes:
        flux_scale: Dimensionless factor on the local zodi alone, default one.
    """

    flux_scale: Array = eqx.field(default=1.0, converter=jnp.asarray, kw_only=True)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Table 19 color breakpoints [nm]; the spectrum is piecewise in them.

        Returns:
            Breakpoint wavelengths [nm], a NumPy array on the host.
        """
        return um_to_nm(TABLE19_WAVELENGTH_UM)

    def _photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr, converted from zodi's per-micron output."""
        if sky_frame.ra_deg is None:
            raise ValueError(
                "LeinertZodi depends on the sightline, but the scene has no sky "
                "position; build it with Scene.from_classical(..., ra_deg=, "
                "dec_deg=) or use a sightline-independent model such as AYOZodi"
            )
        latitude_deg, longitude_deg = ecliptic_pointing(
            sky_frame.ra_deg, sky_frame.dec_deg, sky_frame.epoch, time_d
        )
        return (
            zodi.specific_intensity(
                longitude_deg,
                latitude_deg,
                jnp.asarray(wavelength_nm),
                photon_units=True,
                anchor="table19",
            )
            / 1000.0
        )


@final
class PrecomputedZodi(AbstractLocalZodi):
    """Linearly interpolated photon radiance supplied in per-steradian units.

    The sky frame and time are accepted and ignored. Values outside the
    supplied wavelength support are NaN.

    The table is an ``optixstuff.spectral.SpectralTable``: its knot digest is
    computed at construction, and :meth:`node_radiance` reduces the table on a
    band rule with the operator a ``Resolution`` holds for it.

    Attributes:
        wavelengths_nm: Wavelengths of the table [nm], increasing, shape ``(N,)``.
        radiance_phot_per_s_m2_nm_sr: Photon radiance at those wavelengths
            [ph s^-1 m^-2 nm^-1 sr^-1], shape ``(N,)``.
        flux_scale: Dimensionless factor on the local zodi alone.
        knot_digest: Static digest of the wavelength knots, or None for a table
            constructed under a trace.
        spectral_degree: Polynomial degree between knots: one, linear.
    """

    wavelengths_nm: Array = eqx.field(converter=jnp.asarray)
    radiance_phot_per_s_m2_nm_sr: Array = eqx.field(converter=jnp.asarray)
    flux_scale: Array = eqx.field(converter=jnp.asarray)
    knot_digest: str | None = eqx.field(static=True)
    spectral_degree: ClassVar[int] = 1

    def __init__(
        self,
        wavelengths_nm: Array,
        radiance_phot_per_s_m2_nm_sr: Array,
        flux_scale: Array | float = 1.0,
    ):
        """Store a photon radiance table in ph s^-1 m^-2 nm^-1 sr^-1.

        Args:
            wavelengths_nm: Wavelengths of the table [nm], increasing, shape ``(N,)``.
            radiance_phot_per_s_m2_nm_sr: Photon radiance at those wavelengths
                [ph s^-1 m^-2 nm^-1 sr^-1], shape ``(N,)``.
            flux_scale: Dimensionless factor on the local zodi alone, default one.
        """
        self.flux_scale = jnp.asarray(flux_scale)
        self.wavelengths_nm = jnp.asarray(wavelengths_nm)
        self.knot_digest = table_digest(self.wavelengths_nm, "linear")
        self.radiance_phot_per_s_m2_nm_sr = jnp.asarray(radiance_phot_per_s_m2_nm_sr)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the tabulated radiance wavelengths in nm on the host.

        Returns:
            The table wavelengths [nm], a NumPy array of shape ``(N,)``.
        """
        return np.asarray(self.wavelengths_nm)

    def _photon_radiance(
        self, wavelength_nm: Array, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Photon radiance per sr; the sightline is ignored, wavelength-shaped."""
        wave = jnp.asarray(wavelength_nm)
        return jnp.interp(
            wave,
            self.wavelengths_nm,
            self.radiance_phot_per_s_m2_nm_sr,
            left=jnp.nan,
            right=jnp.nan,
        )

    def spectral_interpolant(self):
        """Return the linear interpolation of a radiance table on the knots.

        Returns:
            A function ``f(values, wavelength_nm)``, linear in values of shape
            ``(N,)``.
        """
        knots = self.spectral_breaks_nm()

        def interpolant(values, wavelength_nm):
            return jnp.interp(wavelength_nm, knots, values)

        return interpolant

    def node_radiance(
        self, quadrature, resolution, sky_frame: SkyFrame, time_d: Array
    ) -> Array:
        """Node-effective radiance from this table's operator, times ``flux_scale``.

        Args:
            quadrature: Band rule from ``optixstuff.spectral.Quadrature.build``.
            resolution: ``optixstuff.spectral.Resolution`` holding a rule for
                this table on ``quadrature``.
            sky_frame: The scene's sky frame; ignored.
            time_d: Observation time, days from ``sky_frame.epoch``; ignored.

        Returns:
            Photon radiance [ph s^-1 m^-2 nm^-1 sr^-1], shape ``(W, Q)``.
        """
        rule = resolution.rule(quadrature, self)
        if rule.kind == "table":
            values = rule.apply(self.radiance_phot_per_s_m2_nm_sr)
        else:
            points = quadrature.nodes_nm if rule.kind == "identity" else rule.fine_nm
            values = rule.apply(self._photon_radiance(points, sky_frame, time_d))
        return self.flux_scale * values

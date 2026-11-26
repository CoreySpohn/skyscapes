"""Profile-dependent calibration of scattered dust amplitudes."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import JOHNSON_V_VEGA_ZERO_POINT_JY, arcsec2rad, pc2AU
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from hwoutils.precision import like
from optixstuff.spectral import Band, same_wavelength
from zodi import DensityDust, SurfaceBrightnessDust

from skyscapes.habitable_zone import eeid_AU

# Half-width of the Johnson V calibration top hat: 506 to 594 nm around the
# 550.0 nm the Johnson zero point is keyed on.
_V_HALF_WIDTH_NM = 44.0


def default_calibration_band(passband):
    """Return the calibration band a passband gets when none is declared.

    Args:
        passband: Calibration passband of the amplitude.

    Returns:
        A top hat from 506 to 594 nm for Johnson V centred on 550.0 nm, else
        None: no other passband has a default band.
    """
    if passband.name == "V" and same_wavelength(passband.center_nm, 550.0):
        return Band(
            passband.center_nm - _V_HALF_WIDTH_NM, passband.center_nm + _V_HALF_WIDTH_NM
        )
    return None


def calibration_band(amplitude, profile):
    """Return the band the amplitude's calibration averages over, or None.

    Args:
        amplitude: ``SurfaceBrightnessDust`` amplitude.
        profile: Parametric disk; its ``calibration_band`` overrides the
            default of the amplitude's passband.

    Returns:
        The declared or default ``optixstuff.spectral.Band``; None when there is
        neither, so the zero point is the monochromatic value at the center.
    """
    declared = getattr(profile, "calibration_band", None)
    if declared is not None:
        return declared
    return default_calibration_band(amplitude.passband)


def zero_magnitude_photon_flux_density(passband, *, zero_point_jy=None, band=None):
    """Photon zero point at the calibration band's center, or its mean over a band.

    AB uses hwoutils' exact AB definition. For Johnson V (Vega, 550 nm)
    we adopt 3636 Jy as the monochromatic calibration convention; other
    Vega bands require an explicit zero_point_jy. The zero point is a flat
    flux density in Jy, F0_ph = F0_Jy * Jy / (h * lambda_m) * nm2m, so its mean
    over a band is F0_Jy * Jy / h times the mean of 1 / lambda, which is
    ln(high / low) / (high - low) in closed form. A magnitude is a band
    quantity, and the host star is averaged over the same band.

    Args:
        passband: Calibration band, giving the magnitude system, name and center
            wavelength.
        zero_point_jy: Monochromatic flux density of a zero-magnitude source
            [Jy]. ``None`` uses the AB definition, or 3636 Jy for Johnson V.
        band: ``optixstuff.spectral.Band`` to average over; ``None`` returns the
            value at ``passband.center_nm``.

    Returns:
        Photon flux density of a zero-magnitude source [ph s^-1 m^-2 nm^-1]: at
        the band center, or the mean over ``band``.
    """
    if zero_point_jy is None:
        if passband.magnitude_system == "ab":
            zero_point_jy = mag_to_flux_jy(0.0)
        elif passband.name == "V" and same_wavelength(passband.center_nm, 550.0):
            zero_point_jy = JOHNSON_V_VEGA_ZERO_POINT_JY
        else:
            raise ValueError("a non-V Vega calibration requires zero_point_jy")
    if band is None:
        return jy_to_photons_per_nm_per_m2(zero_point_jy, passband.center_nm)
    if band.width_nm == 0:
        return jy_to_photons_per_nm_per_m2(zero_point_jy, band.low_nm)
    mean_inverse_nm = np.log(band.high_nm / band.low_nm) / band.width_nm
    return jy_to_photons_per_nm_per_m2(zero_point_jy, 1.0 / mean_inverse_nm)


def check_amplitude(amplitude, profile):
    """Reject incompatible amplitude families, profiles and viewing metadata.

    Args:
        amplitude: Dust amplitude to validate, either a ``DensityDust`` or a
            ``SurfaceBrightnessDust``.
        profile: Parametric disk the amplitude is paired with; supplies the
            ``profile_id`` and ``zero_point_jy`` that are checked.
    """
    if isinstance(amplitude, DensityDust):
        if amplitude.profile_id != profile.profile_id:
            raise ValueError(f"profile_id must be {profile.profile_id!r}")
    elif isinstance(amplitude, SurfaceBrightnessDust):
        if amplitude.viewing.inclination_deg != 0.0:
            raise ValueError("surface-brightness calibration must be face-on")
        if amplitude.reference_star is None:
            raise ValueError("surface-brightness calibration needs a reference_star")
        radius = amplitude.reference_radius_au
        if radius != "eeid" and radius <= 0:
            raise ValueError("reference_radius_au must be positive")
        band = calibration_band(amplitude, profile)
        if band is not None:
            if not isinstance(band, Band):
                raise TypeError("calibration_band must be an optixstuff Band")
            if not same_wavelength(band.center_nm, amplitude.passband.center_nm):
                raise ValueError(
                    f"calibration_band is centred on {band.center_nm} nm but the "
                    f"passband center is {amplitude.passband.center_nm} nm"
                )
        elif amplitude.luminosity_rule == "bolometric":
            raise ValueError(
                "the bolometric rule averages the host star over a band: pass "
                "calibration_band for a passband other than Johnson V"
            )
        zero_magnitude_photon_flux_density(
            amplitude.passband, zero_point_jy=profile.zero_point_jy, band=band
        )
    else:
        raise TypeError("amplitude must be DensityDust or SurfaceBrightnessDust")


def density_level(amplitude, profile, *, star, dist_pc, time_d=0.0):
    """Return the multiplier of this profile's unit scattering density.

    With all lengths in AU, let J(R,lambda) = Ag(lambda) integral
    rho(R,z) p(z/sqrt(R^2+z^2),lambda)/(R^2+z^2) dz. The phase p is
    normalized per sr, rho is a declared unit profile (extinction per AU),
    and J has units AU^-2 sr^-1. Incident stellar illumination is
    Phi_star(lambda,d) * (d_pc*pc2AU)^2 / r_AU^2. Thus rendering gives
    L = level_density * Phi_star * (d_pc*pc2AU)^2 * J, in
    ph s^-1 m^-2 nm^-1 sr^-1, with no pixel solid angle in this expression.

    Evaluate J once face-on at R_ref (sqrt(reference L) for 'eeid').
    The target is amplitude.flux_ratio(host magnitude, host L, R_ref)
    * F0_ph / arcsec2rad^2. Dividing this by Phi_star(ref,d)*d_AU^2*J
    chooses the density multiplier. The host magnitude is inferred from
    that same photon flux and zero point: M=-2.5 log10(Phi/F0)-5 log10(d/10).
    Consequently the band rule cancels the host flux from the multiplier, so
    it does not read the star; the bolometric rule replaces its band luminosity
    by host L exactly once and divides by the star's mean over the calibration
    band (``star.calibration_flux_density``), the same band over which the zero
    point is averaged.
    Subsequent rendering supplies stellar colour and epoch variation, and the
    star's ``flux_scale``, which the calibration does not see.

    For the solar V reference at 10 pc, level=3 and mag=22 give
    3*10^(-0.4*22)=4.7546796e-9 per arcsec^2 (20.8071969 mag/arcsec^2).
    Dividing by 10^(-0.4*4.83) and multiplying by (0.010)^2 gives
    4.0655682e-11 host contrast per 10 mas pixel. DensityDust is already
    a profile multiplier and passes through without host rescaling.

    Args:
        amplitude: Dust amplitude, either a ``DensityDust`` or a
            ``SurfaceBrightnessDust``.
        profile: Parametric disk whose unit scattering density is scaled.
        star: Host star; read only under the bolometric rule, through its
            ``calibration_flux_density`` and ``luminosity_Lsun``.
        dist_pc: Distance of the system from the observer [pc].
        time_d: Time of the calibration in days from the scene epoch; default 0.

    Returns:
        Dimensionless multiplier of the profile's unit scattering density.
    """
    check_amplitude(amplitude, profile)
    if isinstance(amplitude, DensityDust):
        level = amplitude.level
        return (
            jnp.asarray(level)
            if hasattr(level, "dtype")
            else like(level, profile.flux_scale)
        )
    dist_pc = jnp.asarray(dist_pc, dtype=jnp.result_type(dist_pc, profile.flux_scale))
    wave = like(amplitude.passband.center_nm, dist_pc)
    band = calibration_band(amplitude, profile)
    zero = like(
        zero_magnitude_photon_flux_density(
            amplitude.passband, zero_point_jy=profile.zero_point_jy, band=band
        ),
        wave,
    )
    if amplitude.luminosity_rule == "bolometric":
        # The star's own mean over the calibration band, without its
        # ``flux_scale``: the scale is applied at render time, so a disk defined
        # relative to the star follows it instead of cancelling it.
        flux = star.calibration_flux_density(band, time_d, dist_pc=dist_pc)
        luminosity = star.luminosity_Lsun
    else:
        # The host magnitude is built from this same flux, which therefore
        # cancels from the level: the star is not read.
        flux = luminosity = like(1.0, wave)
    absolute_mag = -2.5 * jnp.log10(flux / zero) - 5.0 * jnp.log10(dist_pc / 10.0)
    radius = amplitude.reference_radius_au
    if radius == "eeid":
        radius = eeid_AU(like(amplitude.reference_star.luminosity_lsun, wave))
    radius = like(radius, wave)
    zero_offset = like(0.0, wave)
    integral = profile._scattered_integral(
        wave, radius, zero_offset, zero_offset, zero_offset
    )
    integral = eqx.error_if(
        integral,
        (integral <= 0) | ~jnp.isfinite(integral),
        "profile must scatter at the calibration radius and wavelength",
    )
    target = (
        amplitude.flux_ratio(
            host_band_abs_mag=absolute_mag, host_luminosity_lsun=luminosity, r_au=radius
        )
        * zero
        / arcsec2rad**2
    )
    return target / (flux * (dist_pc * pc2AU) ** 2 * integral)


def surface_brightness_level(
    amplitude, profile, *, calibration, star, dist_pc, time_d=0.0
):
    """Invert density_level using a SurfaceBrightnessDust metadata template.

    The template's level is ignored: linearity gives level_SB = level_rho /
    density_level(template with level=1). Both directions use the identical
    face-on integral, so their round trip is limited only by floating point.
    A template is required because DensityDust cannot encode luminosity_rule
    or mag_per_arcsec2. No conversion exists without an explicit profile.

    Args:
        amplitude: ``DensityDust`` amplitude to convert.
        profile: Parametric disk whose unit scattering density is scaled.
        calibration: ``SurfaceBrightnessDust`` template that supplies the
            luminosity rule, reference radius and brightness metadata.
        star: Host star, supplying the photon flux used in the calibration.
        dist_pc: Distance of the system from the observer [pc].
        time_d: Time of the calibration in days from the scene epoch; default 0.

    Returns:
        Dimensionless surface-brightness level, the density level divided by the
        density level of the template at level one.
    """
    if not isinstance(amplitude, DensityDust):
        raise TypeError("amplitude must be DensityDust")
    if not isinstance(calibration, SurfaceBrightnessDust):
        raise TypeError("calibration must be SurfaceBrightnessDust")
    check_amplitude(amplitude, profile)
    return jnp.asarray(amplitude.level) / density_level(
        replace(calibration, level=1.0),
        profile,
        star=star,
        dist_pc=dist_pc,
        time_d=time_d,
    )

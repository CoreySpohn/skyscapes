"""Profile-dependent calibration of scattered dust amplitudes."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import JOHNSON_V_VEGA_ZERO_POINT_JY, arcsec2rad, pc2AU
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from zodi import DensityDust, SurfaceBrightnessDust

from skyscapes.habitable_zone import eeid_AU


def zero_magnitude_photon_flux_density(passband, *, zero_point_jy=None):
    """Monochromatic photon zero point at the calibration band's center.

    AB uses hwoutils' exact AB definition. For Johnson V (Vega, 550 nm)
    we adopt 3636 Jy as the monochromatic calibration convention; other
    Vega bands require an explicit zero_point_jy. This is a zero-point
    choice, not an integration through a filter: zodi.Passband specifies
    only a center. F0_ph = F0_Jy * Jy / (h * lambda_m) * nm2m.
    """
    if zero_point_jy is None:
        if passband.magnitude_system == "ab":
            zero_point_jy = mag_to_flux_jy(0.0)
        elif passband.name == "V" and passband.center_nm == 550.0:
            zero_point_jy = JOHNSON_V_VEGA_ZERO_POINT_JY
        else:
            raise ValueError("a non-V Vega calibration requires zero_point_jy")
    return jy_to_photons_per_nm_per_m2(zero_point_jy, passband.center_nm)


def check_amplitude(amplitude, profile):
    """Reject incompatible amplitude families, profiles and viewing metadata."""
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
        zero_magnitude_photon_flux_density(
            amplitude.passband, zero_point_jy=profile.zero_point_jy
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
    Consequently the band rule cancels the host flux from the multiplier;
    the bolometric rule replaces its band luminosity by host L exactly once.
    Subsequent rendering supplies stellar colour and epoch variation.

    For the solar V reference at 10 pc, level=3 and mag=22 give
    3*10^(-0.4*22)=4.7546796e-9 per arcsec^2 (20.8071969 mag/arcsec^2).
    Dividing by 10^(-0.4*4.83) and multiplying by (0.010)^2 gives
    4.0655682e-11 host contrast per 10 mas pixel. DensityDust is already
    a profile multiplier and passes through without host rescaling.
    """
    check_amplitude(amplitude, profile)
    if isinstance(amplitude, DensityDust):
        return jnp.asarray(amplitude.level)
    wave = amplitude.passband.center_nm
    zero = zero_magnitude_photon_flux_density(
        amplitude.passband, zero_point_jy=profile.zero_point_jy
    )
    flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
    absolute_mag = -2.5 * jnp.log10(flux / zero) - 5.0 * jnp.log10(dist_pc / 10.0)
    luminosity = (
        star.luminosity_Lsun if amplitude.luminosity_rule == "bolometric" else 1.0
    )
    radius = amplitude.reference_radius_au
    if radius == "eeid":
        radius = eeid_AU(amplitude.reference_star.luminosity_lsun)
    integral = profile._scattered_integral(wave, radius, 0.0, 0.0, 0.0)
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

"""Habitable-zone bounds for a host star.

Two definitions are provided.

**Earth-equivalent insolation distance (EEID) scaling.** The exoEarth
candidate zone of Stark et al. (2019, JATIS 5, 024009, Sec. 2.4) spans
0.95 to 1.67 AU for a solar twin and scales with the square root of the
stellar bolometric luminosity, so its edges are fixed in units of the EEID,
``a_EEID = sqrt(L / L_sun)`` AU. Stark et al. adopt the conservative zone of
Kopparapu et al. (2013, ApJ 765, 131; 2014, ApJ 787, L29) with the inward
revision of the inner edge suggested by 3D climate models; the edges
coincide with the water-loss and maximum-greenhouse limits of Kasting et
al. (1993, Icarus 101, 108) for the Sun. The optimistic edges, 0.75 and
1.77 EEID, are the recent-Venus and early-Mars limits for the Sun of
Kopparapu et al. (2013). These edges depend on luminosity only;
this is the definition used for exoEarth-candidate yields.

**Effective-flux limits.** Kopparapu et al. (2014, Table 1) give
the stellar flux at each limit, relative to the flux at Earth today, as a
quartic in the host effective temperature. The distance of a limit is then
``sqrt((L / L_sun) / S_eff)`` AU. The fits hold for
2600 K <= Teff <= 7200 K; values outside that range are extrapolations and
are not clipped.

Every function is a pure ``jax.numpy`` expression that broadcasts over its
array arguments, so it runs under ``jit``, ``vmap`` and ``grad``.
"""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

HZ_INNER_EEID = 0.95
"""Inner edge of the conservative zone in EEID (Stark et al. 2019)."""

HZ_OUTER_EEID = 1.67
"""Outer edge of the conservative zone in EEID (Stark et al. 2019)."""

OPTIMISTIC_HZ_INNER_EEID = 0.75
"""Inner edge of the optimistic zone in EEID: recent Venus (Kopparapu et al. 2013)."""

OPTIMISTIC_HZ_OUTER_EEID = 1.77
"""Outer edge of the optimistic zone in EEID: early Mars (Kopparapu et al. 2013)."""

KOPPARAPU_REFERENCE_TEFF_K = 5780.0
"""Effective temperature that the Kopparapu et al. (2014) polynomial is centered on."""

KOPPARAPU_SEFF_COEFFICIENTS = {
    "recent_venus": (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runaway_greenhouse": (1.107, 1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "early_mars": (0.32, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
}
"""Coefficients ``(S_eff_sun, a, b, c, d)`` per limit, for a 1 Earth-mass planet.

From Table 1 of Kopparapu et al. (2014). The conservative zone runs from the
runaway-greenhouse limit to the maximum-greenhouse limit, the optimistic zone
from the recent-Venus limit to the early-Mars limit.
"""


def eeid_au(luminosity_lsun: ArrayLike) -> Array:
    """Earth-equivalent insolation distance of a star.

    The distance at which a planet receives the bolometric flux the Earth
    receives from the Sun, ``sqrt(L / L_sun)`` AU.

    Args:
        luminosity_lsun: Stellar bolometric luminosity in solar luminosities.

    Returns:
        The EEID in AU, with the shape of ``luminosity_lsun``.

    Example::

        eeid_au(4.0)  # 2.0 AU
    """
    return jnp.sqrt(jnp.asarray(luminosity_lsun))


def habitable_zone_au(
    luminosity_lsun: ArrayLike, optimistic: bool = False
) -> tuple[Array, Array]:
    """Inner and outer habitable-zone edges scaled by the EEID.

    The conservative zone is 0.95 to 1.67 EEID, the exoEarth-candidate zone
    of Stark et al. (2019); the optimistic zone is 0.75 to 1.77 EEID, the
    recent-Venus and early-Mars limits of Kopparapu et al. (2013). Both scale
    with ``sqrt(L / L_sun)`` and do not depend on the host temperature.

    Args:
        luminosity_lsun: Stellar bolometric luminosity in solar luminosities.
        optimistic: Return the optimistic zone instead of the conservative
            one. A Python bool, static under ``jit``.

    Returns:
        ``(inner_au, outer_au)``, each with the shape of ``luminosity_lsun``.

    Example::

        inner_au, outer_au = habitable_zone_au(1.0)  # 0.95, 1.67
        inner_au, outer_au = jax.vmap(habitable_zone_au)(luminosities_lsun)
    """
    if optimistic:
        inner_eeid, outer_eeid = OPTIMISTIC_HZ_INNER_EEID, OPTIMISTIC_HZ_OUTER_EEID
    else:
        inner_eeid, outer_eeid = HZ_INNER_EEID, HZ_OUTER_EEID
    eeid = eeid_au(luminosity_lsun)
    return inner_eeid * eeid, outer_eeid * eeid


def kopparapu_effective_flux(teff_k: ArrayLike, limit: str) -> Array:
    """Stellar flux at a habitable-zone limit, relative to the flux at Earth.

    Evaluates the parametric fit of Kopparapu et al. (2014),
    ``S_eff = S_eff_sun + a T + b T^2 + c T^3 + d T^4`` with
    ``T = Teff - 5780 K``, using the 1 Earth-mass coefficients of their
    Table 1. Valid for 2600 K <= Teff <= 7200 K.

    Args:
        teff_k: Stellar effective temperature in K.
        limit: One of ``"recent_venus"``, ``"runaway_greenhouse"``,
            ``"maximum_greenhouse"`` or ``"early_mars"``. A Python string,
            static under ``jit`` (bind it in a closure or pass it through
            ``static_argnames``).

    Returns:
        The effective flux, dimensionless, with the shape of ``teff_k``.
    """
    s_sun, a, b, c, d = KOPPARAPU_SEFF_COEFFICIENTS[limit]
    t = jnp.asarray(teff_k) - KOPPARAPU_REFERENCE_TEFF_K
    return s_sun + a * t + b * t**2 + c * t**3 + d * t**4


def kopparapu_habitable_zone_au(
    luminosity_lsun: ArrayLike, teff_k: ArrayLike, optimistic: bool = False
) -> tuple[Array, Array]:
    """Inner and outer habitable-zone edges from the effective-flux limits.

    Each edge is ``sqrt((L / L_sun) / S_eff(Teff))`` AU (Kopparapu et al.
    2014). The conservative zone runs from the runaway-greenhouse
    limit to the maximum-greenhouse limit; the optimistic zone from the
    recent-Venus limit to the early-Mars limit. For the Sun these give
    0.95 to 1.68 AU and 0.75 to 1.77 AU. Valid for 2600 K <= Teff <= 7200 K.

    Args:
        luminosity_lsun: Stellar bolometric luminosity in solar luminosities.
        teff_k: Stellar effective temperature in K, broadcast against
            ``luminosity_lsun``.
        optimistic: Return the optimistic zone instead of the conservative
            one. A Python bool, static under ``jit``.

    Returns:
        ``(inner_au, outer_au)``, each with the broadcast shape of the inputs.

    Example::

        inner_au, outer_au = kopparapu_habitable_zone_au(0.08, 3800.0)
    """
    if optimistic:
        inner_limit, outer_limit = "recent_venus", "early_mars"
    else:
        inner_limit, outer_limit = "runaway_greenhouse", "maximum_greenhouse"
    luminosity_lsun = jnp.asarray(luminosity_lsun)
    inner_au = jnp.sqrt(luminosity_lsun / kopparapu_effective_flux(teff_k, inner_limit))
    outer_au = jnp.sqrt(luminosity_lsun / kopparapu_effective_flux(teff_k, outer_limit))
    return inner_au, outer_au


__all__ = [
    "HZ_INNER_EEID",
    "HZ_OUTER_EEID",
    "KOPPARAPU_REFERENCE_TEFF_K",
    "KOPPARAPU_SEFF_COEFFICIENTS",
    "OPTIMISTIC_HZ_INNER_EEID",
    "OPTIMISTIC_HZ_OUTER_EEID",
    "eeid_au",
    "habitable_zone_au",
    "kopparapu_effective_flux",
    "kopparapu_habitable_zone_au",
]

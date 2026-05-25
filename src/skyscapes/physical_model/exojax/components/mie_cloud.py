"""Mie-scattering cloud component.

A drop-in replacement for :class:`GrayCloud` that uses ExoJAX's
``OpaMie`` to pre-compute wavelength-dependent single-scattering
albedo and asymmetry parameter for a chosen condensate (e.g. water,
water-ice, NH3) and particle size distribution. The cloud's vertical
distribution and total optical depth remain fittable PyTree leaves;
the wavelength dependence of ssa(lambda) and g(lambda) is fixed at
engine-build time.

ExoJAX's ``PdbCloud`` triggers a one-time download of the chosen
condensate's refractive-index data (cached under
``./.database/particulates/virga/``) on first use, similar to the
HITRAN line-list downloads.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.integrate
from jaxtyping import Array

# PyMieScatt (loaded lazily by ExoJAX inside ``generate_miegrid``) calls
# ``scipy.integrate.trapz``, which was removed in SciPy 1.14 (renamed
# to ``trapezoid``). Patch it back before any PyMieScatt-touching
# ExoJAX code runs.
if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from exojax.database.pardb import PdbCloud
from exojax.opacity.opacont import OpaMie

from .base import AbstractClouds, Contribution
from .clouds import DEFAULT_CLOUD_LOG_SIGMA


class MieCloud(AbstractClouds):
    """Single-layer cloud with Mie-scattering optical properties.

    The total cloud scattering optical depth is distributed vertically
    via a softmax-Gaussian in log-pressure (same as
    :class:`GrayCloud`). What's different is that the
    single-scattering albedo and asymmetry parameter come from
    pre-computed Mie cross-sections rather than being scalar
    constants -- they vary with wavelength according to the
    condensate's refractive index n(lambda) + k(lambda).

    Attributes (PyTree leaves, fittable):
        log_pressure_bar: Log10 cloud-deck pressure [bar], shape ``(K,)``.
        log_opt_depth: Log10 of the vertically-integrated cloud
            extinction optical depth, shape ``(K,)``.

    Pre-computed Mie quantities (built at engine time, shared across
    planets):
        ssa_grid: Single-scattering albedo on the wavenumber grid,
            shape ``(n_nu,)``. ``sigma_scattering / sigma_extinction``.
        g_grid: Asymmetry parameter on the wavenumber grid,
            shape ``(n_nu,)``.

    Static config:
        condensate: Condensate name (``"H2O"``, ``"H2O_ice"``,
            ``"MgSiO3"``, etc.). Matters for the repr; the actual
            Mie params are baked into ssa_grid and g_grid.
        rg_um: Geometric mean particle radius [um].
        sigmag: Geometric standard deviation of the lognormal size
            distribution.
        log_sigma: Cloud-deck vertical Gaussian half-width in
            log10(pressure) [dex].
    """

    log_pressure_bar: Array
    log_opt_depth: Array
    ssa_grid: Array
    g_grid: Array
    condensate: str = eqx.field(static=True, default="H2O")
    rg_um: float = eqx.field(static=True, default=10.0)
    sigmag: float = eqx.field(static=True, default=2.0)
    log_sigma: float = eqx.field(static=True, default=DEFAULT_CLOUD_LOG_SIGMA)

    def compute(
        self,
        log_pressure_bar_scalar: Array,
        log_opt_depth_scalar: Array,
        pressure: Array,
        n_nu: int,
    ) -> Contribution:
        """Mie-cloud contribution: gray-depth distribution, spectral ssa/g."""
        n_layers = pressure.shape[0]
        log_p_layers = jnp.log10(pressure)
        log_w = -0.5 * ((log_p_layers - log_pressure_bar_scalar) / self.log_sigma) ** 2
        cloud_weights = jax.nn.softmax(log_w)
        tau_total = 10.0**log_opt_depth_scalar
        dtau_layer = tau_total * cloud_weights  # (n_layers,)
        dtau = jnp.broadcast_to(dtau_layer[:, None], (n_layers, n_nu))

        # Wavelength-dependent ssa(lambda), g(lambda) broadcast across
        # layers. Unlike GrayCloud's scalar ssa/g, Mie clouds vary the
        # scattering properties across the spectrum.
        ssa_per_nu = jnp.broadcast_to(self.ssa_grid[None, :], dtau.shape)
        g_per_nu = jnp.broadcast_to(self.g_grid[None, :], dtau.shape)
        dtau_scatter = ssa_per_nu * dtau
        g_weighted_num = g_per_nu * dtau_scatter
        return Contribution(
            dtau_total=dtau,
            dtau_scatter=dtau_scatter,
            g_weighted_num=g_weighted_num,
        )


def build_mie_cloud(
    *,
    nu_grid: Array,
    log_pressure_bar: Array,
    log_opt_depth: Array,
    condensate: str = "H2O",
    rg_um: float = 10.0,
    sigmag: float = 2.0,
    log_sigma: float = DEFAULT_CLOUD_LOG_SIGMA,
) -> MieCloud:
    """Build a Mie cloud component by pre-computing Mie params.

    On first use for a given condensate this triggers two downloads /
    computations cached under ``./.database/particulates/virga/``:

      - **Refractive-index file** (small, fetched from Zenodo).
      - **Mie-grid lookup table** (built locally via PyMieScatt; takes
        a couple of minutes per condensate). Subsequent calls reuse
        the cache.

    Args:
        nu_grid: Wavenumber grid [cm^-1] from
            :func:`build_exojax_engines`.
        log_pressure_bar: Per-planet log10 cloud pressure [bar].
        log_opt_depth: Per-planet log10 total cloud extinction tau.
        condensate: Condensate name. ExoJAX/Virga ships e.g.
            ``"H2O"``, ``"H2O_ice"``, ``"NH3"``, ``"MgSiO3"``,
            ``"Mg2SiO4"``, ``"Fe"``, ``"KCl"``, ``"Na2S"``,
            ``"ZnS"``, ``"MnS"``, ``"Cr"``, ``"Al2O3"``,
            ``"TiO2"``.
        rg_um: Mean particle radius [um] of the lognormal size
            distribution.
        sigmag: Geometric standard deviation of the size distribution.
        log_sigma: Vertical-distribution Gaussian half-width [dex].

    Returns:
        A :class:`MieCloud` instance ready to drop into
        :class:`ExoJaxPhysicalModel`.
    """
    pdb = PdbCloud(condensate=condensate)
    if not pdb.miegrid_path.exists():
        pdb.generate_miegrid()
    pdb.load_miegrid()
    opa_mie = OpaMie(pdb=pdb, nu_grid=nu_grid)
    rg_cm = rg_um * 1.0e-4
    sig_ext, sig_sca, g_arr = opa_mie.mieparams_vector(rg_cm, sigmag)
    # ssa = sigma_scattering / sigma_extinction; clipped to avoid
    # 0/0 when the cross-section is zero outside coverage.
    ssa_grid = jnp.where(sig_ext > 0, sig_sca / sig_ext, 0.0)
    return MieCloud(
        log_pressure_bar=log_pressure_bar,
        log_opt_depth=log_opt_depth,
        ssa_grid=ssa_grid,
        g_grid=g_arr,
        condensate=condensate,
        rg_um=rg_um,
        sigmag=sigmag,
        log_sigma=log_sigma,
    )

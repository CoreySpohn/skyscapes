"""ExoJaxEmissionModel: composition-based thermal-emission planet model.

The thermal-infrared counterpart of
:class:`~skyscapes.physical_model.exojax.physical_model.ExoJaxPhysicalModel`.
It shares that model's species records, opacity engines, mixing-ratio
profiles and :class:`~.components.absorption.Absorption` component, and
replaces the reflected-light solver with ExoJAX's ``ArtEmisPure``
(intensity-based, pure absorption) plus an opaque emitting ground.

Known omissions:

- Collision-induced absorption (N2-N2, O2-O2, CO2-CO2, ...) is not
  included. It matters mainly in the 8-12 micron window for dense CO2
  atmospheres and is a small effect for an Earth twin.
- Scattering (Rayleigh and clouds) is not included: the atmosphere is a
  cloud-free, purely absorbing, plane-parallel column.
- The planet is treated as a uniformly emitting disk, with no day-night
  contrast and no phase dependence.
- Downwelling radiation reflected by a ground with emissivity below one
  is neglected.

Example::

    import jax.numpy as jnp
    from skyscapes.physical_model.exojax import ExoJaxEmissionModel

    model = ExoJaxEmissionModel.from_default_setup(
        log_mmrs={"H2O": jnp.array([-3.0]), "CO2": jnp.array([-3.4])},
        temperature_coefficients=jnp.array([[285.2, 146.6, 99.7, 23.1, 1.7]]),
        log_gravity_cgs=jnp.array([2.99]),
    )
    flux_jy = model.flux_density_jy(
        wavelength_nm=jnp.linspace(4000.0, 18500.0, 500),
        Rp_Rearth=jnp.array([1.0]),
        distance_pc=10.0,
    )  # shape (500, 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import interpax
import jax.numpy as jnp
from exojax.rt import ArtEmisPure
from exojax.utils.grids import wavenumber_grid
from hwoutils import constants as const
from hwoutils.constants import G_si, Mearth2kg, Rearth2AU, Rearth2m
from hwoutils.conversions import nm_to_wavenumber_cm
from jaxtyping import Array

from .components import (
    Absorption,
    ConstantMmr,
    EmittingSurface,
    MolecularSpecies,
    PolynomialTPProfile,
    build_species_prebuilt,
)
from .components.mmr_profile import AbstractMmrProfile
from .physical_model import _species_at_planet

# Optical depth of the opaque ground layer; exp(-tau / mu) underflows to
# exactly zero for every quadrature angle.
_GROUND_OPTICAL_DEPTH = 1.0e3

# ExoJAX works in cgs; hwoutils is SI.
_M_TO_CM = 100.0
_J_S_TO_ERG_S = 1.0e7  # J and J/K to erg and erg/K
_ERG_PER_S_CM2_PER_HZ_TO_JY = 1.0e-3 / const.Jy

DEFAULT_EMISSION_MOLECULES: tuple[str, ...] = ("H2O", "CO2", "O3", "CH4", "N2O")


def _top_of_atmosphere_flux(
    dtau: Array, source: Array, mus: Array, weights: Array
) -> Array:
    """Emergent flux from layer optical depths and Planck sources.

    The intensity at each quadrature angle is the sum over layers of the
    layer source weighted by the layer's share of the slant transmission
    (``exp(-tau_top / mu) - exp(-tau_bottom / mu)``), then integrated over
    the hemisphere with weights ``2 mu w``. Layers are ordered top to
    bottom; the last layer may be an opaque ground.
    """
    tau = jnp.cumsum(dtau, axis=0)  # (L, n_nu)
    trans = jnp.exp(-tau[None] / mus[:, None, None])  # (M, L, n_nu)
    above = jnp.concatenate([jnp.ones_like(trans[:, :1]), trans[:, :-1]], axis=1)
    intensity = jnp.sum(source[None] * (above - trans), axis=1)  # (M, n_nu)
    return jnp.einsum("m,m,mn->n", 2.0 * mus, weights, intensity)


def log_gravity_cgs_from_mass_radius(mass_Mearth: Array, radius_Rearth: Array) -> Array:
    """Log10 surface gravity [cm/s^2] from mass and radius in Earth units."""
    g_si = G_si * mass_Mearth * Mearth2kg / (radius_Rearth * Rearth2m) ** 2
    return jnp.log10(g_si * _M_TO_CM)


def planck_flux_cgs(nu_cm: Array, temperature_K: Array) -> Array:
    """``pi B_nu(T)`` [erg/s/cm^2/cm^-1] at wavenumber ``nu_cm`` [cm^-1].

    Uses hwoutils constants. ``temperature_K`` of shape ``S`` gives an
    output of shape ``S + nu_cm.shape``.
    """
    h_cgs = const.h * _J_S_TO_ERG_S
    c_cgs = const.c * _M_TO_CM
    k_cgs = const.k_B * _J_S_TO_ERG_S
    x = h_cgs * c_cgs * nu_cm / (k_cgs * jnp.asarray(temperature_K)[..., None])
    return 2.0 * jnp.pi * h_cgs * c_cgs**2 * nu_cm**3 / jnp.expm1(x)


def brightness_temperature(nu_cm: Array, flux_cgs: Array) -> Array:
    """Brightness temperature [K] of a flux ``pi B_nu`` [erg/s/cm^2/cm^-1].

    Inverts the Planck function; ``flux_cgs`` and ``nu_cm`` broadcast.
    """
    h_cgs = const.h * _J_S_TO_ERG_S
    c_cgs = const.c * _M_TO_CM
    k_cgs = const.k_B * _J_S_TO_ERG_S
    intensity = flux_cgs / jnp.pi
    x = 2.0 * h_cgs * c_cgs**2 * nu_cm**3 / intensity
    return h_cgs * c_cgs * nu_cm / k_cgs / jnp.log1p(x)


def build_exojax_emission_engines(
    *,
    molecules: tuple[str, ...] = DEFAULT_EMISSION_MOLECULES,
    wavelength_min_nm: float = 4000.0,
    wavelength_max_nm: float = 18500.0,
    n_wavenumbers: int = 6000,
    n_layers: int = 100,
    pressure_top_bar: float = 1.0e-4,
    surface_pressure_bar: float = 1.0,
    temperature_range_K: tuple[float, float] = (150.0, 350.0),
    databases_dir: str | None = None,
    crit: float = 0.0,
) -> dict:
    """Build the heavy shared engines for thermal emission.

    Same construction pattern as
    :func:`~skyscapes.physical_model.exojax.physical_model.build_exojax_engines`,
    with an ``ArtEmisPure`` engine and HITRAN line lists for every
    molecule (including O3 and SO2, whose reflected-light recipes use
    ultraviolet/visible cross-section tables that do not extend to the
    infrared). The model's bottom layer sits at ``surface_pressure_bar``.

    Args:
        molecules: Molecule names, keys of ``MOLECULE_RECIPES``.
        wavelength_min_nm: Short-wavelength end of the range [nm].
        wavelength_max_nm: Long-wavelength end of the range [nm].
        n_wavenumbers: Points in the wavenumber grid.
        n_layers: Number of atmospheric layers.
        pressure_top_bar: Pressure at the top of the model [bar].
        surface_pressure_bar: Surface pressure, the bottom of the model [bar].
        temperature_range_K: Range over which line opacities are tabulated;
            must bracket the T-P profile.
        databases_dir: HITRAN line-list cache (never downloaded here unless
            the cache is empty).
        crit: Line-strength cutoff passed to ``MdbHitran``.

    Returns:
        Dict with keys ``rt_engine``, ``nu_grid``, ``n_nu`` and
        ``species_prebuilt`` (name to ``{molmass, opa, rayleigh_xs}``).
    """
    if databases_dir is None:
        databases_dir = str(Path.home() / ".cache" / "skyscapes" / "exojax")
    Path(databases_dir).mkdir(parents=True, exist_ok=True)

    nu_max = nm_to_wavenumber_cm(wavelength_min_nm)
    nu_min = nm_to_wavenumber_cm(wavelength_max_nm)
    nu_grid, _, _ = wavenumber_grid(
        nu_min, nu_max, n_wavenumbers, xsmode="premodit", unit="cm-1"
    )
    rt_engine = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=pressure_top_bar,
        pressure_btm=surface_pressure_bar,
        nlayer=n_layers,
    )

    species_prebuilt: dict[str, dict[str, Any]] = {}
    for name in molecules:
        molmass, opa, rayleigh_xs = build_species_prebuilt(
            name=name,
            nu_grid=nu_grid,
            nu_min=float(nu_min),
            nu_max=float(nu_max),
            databases_dir=databases_dir,
            crit=crit,
            use_line_lists=True,
            temperature_range_K=temperature_range_K,
            rayleigh=False,
        )
        species_prebuilt[name] = {
            "molmass": molmass,
            "opa": opa,
            "rayleigh_xs": rayleigh_xs,
        }
    return {
        "rt_engine": rt_engine,
        "nu_grid": nu_grid,
        "n_nu": int(nu_grid.shape[0]),
        "species_prebuilt": species_prebuilt,
    }


class ExoJaxEmissionModel(eqx.Module):
    """Thermal-emission planet model over ExoJAX's ``ArtEmisPure``.

    Per-planet state (PyTree leaves, shape ``(K, ...)``): gravity, the
    species mixing-ratio profiles, the polynomial T-P coefficients and the
    ground temperature and emissivity. The spectrum is the disk-integrated
    flux of a uniformly emitting planet; see the module docstring for the
    omissions.

    Attributes:
        log_gravity_cgs: Log10 surface gravity [cm/s^2], shape ``(K,)``.
        species: Tuple of :class:`MolecularSpecies`. Each holds an HITRAN
            opacity engine built by :func:`build_exojax_emission_engines`.
        tp_profile: :class:`PolynomialTPProfile`.
        surface: :class:`EmittingSurface`.
        absorption: :class:`Absorption` component (shared with the
            reflected-light model).
        rt_engine: ExoJAX ``ArtEmisPure`` instance. Its bottom pressure is
            the surface pressure.
        nu_grid: Wavenumber grid [cm^-1].
        n_nu: Length of ``nu_grid``.
    """

    log_gravity_cgs: Array
    species: tuple[MolecularSpecies, ...]
    tp_profile: PolynomialTPProfile
    surface: EmittingSurface
    absorption: Absorption
    rt_engine: Any = eqx.field(static=True)
    nu_grid: Array
    n_nu: int = eqx.field(static=True)

    @classmethod
    def from_default_setup(
        cls,
        *,
        log_mmrs: dict[str, Array | AbstractMmrProfile],
        temperature_coefficients: Array,
        log_gravity_cgs: Array,
        surface_temperature_K: Array | None = None,
        surface_emissivity: Array | None = None,
        surface_emissivity_spectrum: Array | None = None,
        molecules: tuple[str, ...] | None = None,
        wavelength_min_nm: float = 4000.0,
        wavelength_max_nm: float = 18500.0,
        n_wavenumbers: int = 6000,
        n_layers: int = 100,
        pressure_top_bar: float = 1.0e-4,
        surface_pressure_bar: float = 1.0,
        temperature_range_K: tuple[float, float] = (150.0, 350.0),
        databases_dir: str | None = None,
        crit: float = 0.0,
    ) -> ExoJaxEmissionModel:
        """Build engines and default components in one call.

        Args:
            log_mmrs: Molecule name to per-planet log10 mass-mixing ratio,
                shape ``(K,)`` (constant profile), or to an
                :class:`AbstractMmrProfile`.
            temperature_coefficients: Polynomial T-P coefficients
                ``a_0 ... a_n`` [K] in ``log10 P[bar]``, shape ``(K, n + 1)``.
            log_gravity_cgs: Log10 surface gravity [cm/s^2], ``(K,)``; see
                :func:`log_gravity_cgs_from_mass_radius`.
            surface_temperature_K: Ground temperature [K], ``(K,)``. Defaults
                to the T-P profile evaluated at the surface pressure.
            surface_emissivity: Ground emissivity, ``(K,)``. Defaults to 1.
            surface_emissivity_spectrum: ``(n_nu,)`` emissivity shape.
                Defaults to flat ones.
            molecules: Override the molecule list (default: the keys of
                ``log_mmrs``).
            wavelength_min_nm: See :func:`build_exojax_emission_engines`.
            wavelength_max_nm: See :func:`build_exojax_emission_engines`.
            n_wavenumbers: See :func:`build_exojax_emission_engines`.
            n_layers: See :func:`build_exojax_emission_engines`.
            pressure_top_bar: See :func:`build_exojax_emission_engines`.
            surface_pressure_bar: See :func:`build_exojax_emission_engines`.
            temperature_range_K: See :func:`build_exojax_emission_engines`.
            databases_dir: See :func:`build_exojax_emission_engines`.
            crit: See :func:`build_exojax_emission_engines`.
        """
        if molecules is None:
            molecules = tuple(log_mmrs.keys())
        engines = build_exojax_emission_engines(
            molecules=molecules,
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm,
            n_wavenumbers=n_wavenumbers,
            n_layers=n_layers,
            pressure_top_bar=pressure_top_bar,
            surface_pressure_bar=surface_pressure_bar,
            temperature_range_K=temperature_range_K,
            databases_dir=databases_dir,
            crit=crit,
        )
        species = []
        for name, value in log_mmrs.items():
            pre = engines["species_prebuilt"][name]
            profile = (
                value if isinstance(value, AbstractMmrProfile) else ConstantMmr(value)
            )
            species.append(
                MolecularSpecies(
                    profile=profile,
                    name=name,
                    molmass=pre["molmass"],
                    opa=pre["opa"],
                    rayleigh_xs=pre["rayleigh_xs"],
                )
            )
        return cls.from_engines(
            engines=engines,
            species=tuple(species),
            temperature_coefficients=temperature_coefficients,
            log_gravity_cgs=log_gravity_cgs,
            surface_temperature_K=surface_temperature_K,
            surface_emissivity=surface_emissivity,
            surface_emissivity_spectrum=surface_emissivity_spectrum,
            surface_pressure_bar=surface_pressure_bar,
        )

    @classmethod
    def from_engines(
        cls,
        *,
        engines: dict,
        species: tuple[MolecularSpecies, ...],
        temperature_coefficients: Array,
        log_gravity_cgs: Array,
        surface_temperature_K: Array | None = None,
        surface_emissivity: Array | None = None,
        surface_emissivity_spectrum: Array | None = None,
        surface_pressure_bar: float | None = None,
    ) -> ExoJaxEmissionModel:
        """Assemble a model from prebuilt engines (build once, fit many).

        ``engines`` is the dict returned by
        :func:`build_exojax_emission_engines`. ``surface_pressure_bar``
        defaults to the engine's bottom pressure.
        """
        tp_profile = PolynomialTPProfile(
            coefficients=jnp.atleast_2d(jnp.asarray(temperature_coefficients))
        )
        K = log_gravity_cgs.shape[0]
        rt_engine = engines["rt_engine"]
        n_nu = engines["n_nu"]
        if surface_pressure_bar is None:
            surface_pressure_bar = float(jnp.max(rt_engine.pressure))
        if surface_temperature_K is None:
            surface_temperature_K = jnp.stack(
                [
                    tp_profile.temperature(jnp.asarray(surface_pressure_bar), k)
                    for k in range(K)
                ]
            )
        if surface_emissivity is None:
            surface_emissivity = jnp.ones(K)
        if surface_emissivity_spectrum is None:
            surface_emissivity_spectrum = jnp.ones(n_nu)
        return cls(
            log_gravity_cgs=log_gravity_cgs,
            species=species,
            tp_profile=tp_profile,
            surface=EmittingSurface(
                temperature_K=surface_temperature_K,
                emissivity=surface_emissivity,
                spectrum=surface_emissivity_spectrum,
            ),
            absorption=Absorption(),
            rt_engine=rt_engine,
            nu_grid=engines["nu_grid"],
            n_nu=n_nu,
        )

    def layer_temperature(self, k: int = 0) -> Array:
        """Layer temperatures [K] of planet ``k``, shape ``(n_layers,)``."""
        return self.tp_profile.compute_Tarr(
            self.rt_engine, self.tp_profile.coefficients[k]
        )

    def _emergent_flux_one_planet(self, k: int) -> Array:
        """Emergent flux of planet ``k`` [erg/s/cm^2/cm^-1], shape ``(n_nu,)``."""
        pressure = self.rt_engine.pressure
        gravity = 10.0 ** self.log_gravity_cgs[k]
        temperature = self.layer_temperature(k)
        dtau = self.absorption.compute(
            _species_at_planet(self.species, k),
            temperature,
            pressure,
            gravity,
            self.rt_engine,
        ).dtau_total

        source = planck_flux_cgs(self.nu_grid, temperature)
        ground = self.surface.compute_emissivity(k) * planck_flux_cgs(
            self.nu_grid, self.surface.temperature_K[k]
        )
        # The ground is an extra optically thick bottom layer whose source
        # is the (emissivity-weighted) Planck flux.
        dtau = jnp.concatenate([dtau, jnp.full((1, self.n_nu), _GROUND_OPTICAL_DEPTH)])
        source = jnp.concatenate([source, ground[None, :]])
        return _top_of_atmosphere_flux(
            dtau, source, self.rt_engine.mus, self.rt_engine.weights
        )

    def emergent_flux(self) -> Array:
        """Flux leaving the top of the atmosphere, shape ``(K, n_nu)``.

        Units are erg/s/cm^2/cm^-1 per unit planet surface area, on
        ``nu_grid``. A blackbody at ``T`` gives ``pi B_nu(T)``.
        """
        K = int(self.log_gravity_cgs.shape[0])
        return jnp.stack([self._emergent_flux_one_planet(k) for k in range(K)])

    def brightness_temperature(self) -> Array:
        """Brightness temperature [K] on ``nu_grid``, shape ``(K, n_nu)``."""
        return brightness_temperature(self.nu_grid, self.emergent_flux())

    def flux_density_jy(
        self, wavelength_nm: Array, Rp_Rearth: Array, distance_pc: Array
    ) -> Array:
        """Planet flux density at the observer [Jy], shape ``(W, K)``.

        ``F_nu = F_emergent * (Rp / d)^2``, converted from per wavenumber
        to per frequency. The axis order matches
        :meth:`~skyscapes.physical_model.exojax.physical_model.ExoJaxPhysicalModel.contrast_cube`
        (wavelength first); multiplying that cube's contrast by a stellar
        flux in Jy gives the reflected-light flux on the same axis.

        Args:
            wavelength_nm: Wavelengths [nm], shape ``(W,)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            distance_pc: Distance to the system [pc], scalar or ``(K,)``.
        """
        flux = self.emergent_flux()
        target_nu = nm_to_wavenumber_cm(jnp.atleast_1d(wavelength_nm))
        at_wl = jnp.stack(
            [
                interpax.interp1d(
                    target_nu, self.nu_grid, f, method="cubic", extrap=True
                )
                for f in flux
            ],
            axis=1,
        )  # (W, K)
        f_nu_cgs = at_wl / (const.c * _M_TO_CM)
        rp_over_d = (Rp_Rearth * Rearth2AU) / (distance_pc * const.pc2AU)
        return f_nu_cgs * _ERG_PER_S_CM2_PER_HZ_TO_JY * rp_over_d**2

    def __repr__(self) -> str:
        """One-line summary of species, grid and planet count."""
        species_label = ", ".join(s.name for s in self.species)
        return (
            f"ExoJaxEmissionModel(K={int(self.log_gravity_cgs.shape[0])}, "
            f"species=[{species_label}], n_nu={self.n_nu}, "
            f"n_layers={int(self.rt_engine.pressure.shape[0])})"
        )

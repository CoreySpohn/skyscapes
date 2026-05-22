"""ExoJaxAtmosphere: composition-based reflected-light atmosphere.

Each piece of physics is a swappable component; each molecule is a
self-contained :class:`MolecularSpecies` record. To explore different
atmospheres, swap a component or a species and rebuild.

What's currently included:

  - **Species** (:class:`MolecularSpecies` tuple): variable-length set
    of molecules, each carrying its own per-planet log mass-mixing
    ratio + the shared static data (molar mass, opacity engine,
    Rayleigh cross-section). The biosignature five (H2O, CO2, CH4, O2,
    O3) is preregistered, plus CO, N2O, SO2 for richer atmospheres
    (e.g. the Star+ "Earth through time" set).
  - **Bulk gas** (:class:`BulkGasResidual`): implicit residual gas
    (N2 default, H2/He available) filling 1 - sum(tracked mmrs).
    Contributes only Rayleigh scattering.
  - **Absorption** (:class:`Absorption` component): iterates species,
    sums line-list / cross-section contributions.
  - **Scattering** (:class:`RayleighScattering` or :class:`NullScattering`):
    Rayleigh from tracked species + bulk.
  - **Clouds** (:class:`GrayCloud` / :class:`NoCloud`): single-layer
    gray cloud with softmax-Gaussian vertical distribution.
  - **Surface** (:class:`WavelengthDependentSurface` / :class:`FlatSurface`):
    per-planet albedo scalar x wavelength-dependent reflectivity profile.
  - **T-P profile** (:class:`PowerLawTPProfile`): ``T(P) = T_eq * P^alpha``.

Construction patterns:

- **One-shot**: ``ExoJaxAtmosphere.from_default_setup(...)`` accepts
  ``log_mmrs`` as a dict mapping molecule name to ``(K,)`` array and
  builds engines + default components in one call.
- **Build-once, fit-many** (retrieval-friendly): call
  :func:`build_exojax_engines(molecules=...)` once for the heavy
  per-molecule construction (HITRAN downloads + opacity precompute),
  then assemble many atmospheres per sample.

References:
    Kawahara et al. 2022, ApJS 258, 31 (ExoJAX Paper I)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from exojax.rt import ArtReflectPure
from exojax.utils.grids import wavenumber_grid
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import (
    nm_to_wavenumber_cm,
    spherical_to_geometric_albedo,
)
from jaxtyping import Array

from ..._repr import indent
from ..base import AbstractAtmosphere
from ..cached import PrecomputedReflectivity
from ..lambertian import _lambert_phase
from .components import (
    Absorption,
    AbstractAbsorption,
    AbstractClouds,
    AbstractMmrProfile,
    AbstractScattering,
    AbstractSurface,
    AbstractTPProfile,
    BulkGasResidual,
    ConstantMmr,
    GrayCloud,
    MolecularSpecies,
    PowerLawTPProfile,
    RayleighScattering,
    WavelengthDependentSurface,
    build_bulk_prebuilt,
    build_species_prebuilt,
)

# Default molecule set: the five-molecule biosignature set used in
# most HWO reflected-light studies. Replace via ``molecules=...`` to
# build engines for a different set (e.g. the 8-molecule Star+ "Earth
# through time" composition).
DEFAULT_MOLECULES: tuple[str, ...] = ("H2O", "CO2", "CH4", "O2", "O3")


def build_exojax_engines(
    *,
    molecules: tuple[str, ...] = DEFAULT_MOLECULES,
    bulk_gas: str | None = "N2",
    wavelength_min_nm: float = 400.0,
    wavelength_max_nm: float = 1000.0,
    n_wavenumbers: int = 2000,
    n_layers: int = 100,
    pressure_top_bar: float = 1.0e-5,
    pressure_btm_bar: float = 1.0e0,
    databases_dir: str | None = None,
    crit: float = 0.0,
) -> dict:
    """Build the heavy shared engines and pre-built per-species data.

    Each molecule's opacity engine (line-list LSD tables or cross-
    section interpolant) is built once here and reused across many
    atmosphere instantiations that vary only the per-planet
    ``log_mmrs``. The retrieval-friendly construction pattern is::

        engines = build_exojax_engines(molecules=("H2O", "CO", "O2"))

        def make_atmosphere(per_planet_log_mmrs):
            return ExoJaxAtmosphere.from_default_setup(
                log_mmrs=per_planet_log_mmrs, ..., **engines
            )

    Args:
        molecules: Tuple of molecule names to include. Must all appear
            in :data:`MOLECULE_RECIPES`. Default is the 5-molecule
            biosignature set.
        bulk_gas: Implicit residual gas name (``"N2"``, ``"H2"``,
            ``"He"``), or ``None`` to disable the bulk-gas Rayleigh
            contribution. Default ``"N2"`` for terrestrial atmospheres.
        wavelength_min_nm: Short-wavelength end of the spectral range [nm].
        wavelength_max_nm: Long-wavelength end of the spectral range [nm].
        n_wavenumbers: Number of points in the wavenumber grid.
        n_layers: Number of layers in the plane-parallel RT solver.
        pressure_top_bar: Pressure at the top of the model [bar].
        pressure_btm_bar: Pressure at the bottom of the model [bar].
        databases_dir: ExoJAX line-list cache.
        crit: Line-strength cutoff [cm/(molecule.cm^-2)] passed to
            ``MdbHitran`` (default ``0.0`` = no filtering). Set
            ``crit=1e-26`` for ~25% speedup at HWO contrast levels.

    Returns:
        Dict ready to be ``**``-unpacked into
        :class:`ExoJaxAtmosphere`. Keys: ``rt_engine``, ``nu_grid``,
        ``n_nu``, ``species_prebuilt`` (dict mapping name to
        ``{molmass, opa, rayleigh_xs}``), ``bulk_prebuilt``
        (dict with ``{name, molmass, rayleigh_xs}`` or ``None``).
    """
    if databases_dir is None:
        databases_dir = str(Path.home() / ".cache" / "skyscapes" / "exojax")
    Path(databases_dir).mkdir(parents=True, exist_ok=True)

    nu_max = nm_to_wavenumber_cm(wavelength_min_nm)
    nu_min = nm_to_wavenumber_cm(wavelength_max_nm)
    nu_grid, _, _ = wavenumber_grid(
        nu_min, nu_max, n_wavenumbers, xsmode="premodit", unit="cm-1"
    )

    rt_engine = ArtReflectPure(
        nu_grid=nu_grid,
        pressure_top=pressure_top_bar,
        pressure_btm=pressure_btm_bar,
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
        )
        species_prebuilt[name] = {
            "molmass": molmass,
            "opa": opa,
            "rayleigh_xs": rayleigh_xs,
        }

    if bulk_gas is None:
        bulk_prebuilt = None
    else:
        bulk_molmass, bulk_rayleigh_xs = build_bulk_prebuilt(
            name=bulk_gas, nu_grid=nu_grid
        )
        bulk_prebuilt = {
            "name": bulk_gas,
            "molmass": bulk_molmass,
            "rayleigh_xs": bulk_rayleigh_xs,
        }

    return {
        "rt_engine": rt_engine,
        "nu_grid": nu_grid,
        "n_nu": int(nu_grid.shape[0]),
        "species_prebuilt": species_prebuilt,
        "bulk_prebuilt": bulk_prebuilt,
    }


def _assemble_species(
    log_mmrs: dict[str, Array | AbstractMmrProfile],
    species_prebuilt: dict[str, dict[str, Any]],
) -> tuple[MolecularSpecies, ...]:
    """Combine per-planet log_mmrs with prebuilt opacity data into species.

    Each dict entry can be either a per-planet ``(K,)`` array (treated
    as a constant-mmr profile, the back-compat path) or an
    :class:`AbstractMmrProfile` instance (use as-is for altitude-
    resolved profiles like O3 stratospheric peak or H2O cold trap).
    """
    species = []
    for name, value in log_mmrs.items():
        if name not in species_prebuilt:
            raise KeyError(
                f"Species {name!r} not in prebuilt set "
                f"{sorted(species_prebuilt.keys())}. "
                f"Pass molecules=(...) to build_exojax_engines to include it."
            )
        pre = species_prebuilt[name]
        if isinstance(value, AbstractMmrProfile):
            profile = value
        else:
            profile = ConstantMmr(log_mmr=value)
        species.append(
            MolecularSpecies(
                profile=profile,
                name=name,
                molmass=pre["molmass"],
                opa=pre["opa"],
                rayleigh_xs=pre["rayleigh_xs"],
            )
        )
    return tuple(species)


def _assemble_bulk(
    bulk_prebuilt: dict[str, Any] | None,
) -> BulkGasResidual | None:
    """Construct the bulk-gas residual from prebuilt data, or return None."""
    if bulk_prebuilt is None:
        return None
    return BulkGasResidual(
        name=bulk_prebuilt["name"],
        molmass=bulk_prebuilt["molmass"],
        rayleigh_xs=bulk_prebuilt["rayleigh_xs"],
    )


# Bump when the set of from_default_setup kwargs changes meaning in a
# way that would silently invalidate previously-saved caches.
_CACHE_KEY_VERSION = 1


def _hash_value(value: Any) -> bytes:
    """Stable byte representation of an arbitrary kwarg value.

    Handles JAX/NumPy arrays, dicts (sorted keys), tuples/lists,
    :class:`equinox.Module` instances (via the PyTree protocol), and
    plain scalars. Used by :func:`_cache_key` to build a deterministic
    hash of ``from_default_setup_cached``'s arguments.
    """
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        arr = np.asarray(value)
        return arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()
    if isinstance(value, dict):
        return b"||".join(
            k.encode() + b":" + _hash_value(value[k]) for k in sorted(value.keys())
        )
    if isinstance(value, (tuple, list)):
        return b"||".join(_hash_value(item) for item in value)
    if isinstance(value, eqx.Module):
        leaves, treedef = jax.tree.flatten(value)
        pieces = [
            type(value).__name__.encode(),
            repr(treedef).encode(),
        ]
        for leaf in leaves:
            pieces.append(_hash_value(leaf))
        return b"||".join(pieces)
    if value is None:
        return b"None"
    return repr(value).encode()


def _cache_key(**kwargs: Any) -> str:
    """SHA256 hex digest of the cached factory's kwargs + version."""
    h = hashlib.sha256()
    h.update(f"v{_CACHE_KEY_VERSION}".encode())
    for name in sorted(kwargs.keys()):
        h.update(name.encode())
        h.update(_hash_value(kwargs[name]))
    return h.hexdigest()[:16]


class ExoJaxAtmosphere(AbstractAtmosphere):
    """Composition-based reflected-light atmosphere over ExoJAX's 2-stream RT.

    Per-planet state (PyTree leaves, fittable):
        Rp_Rearth: Planetary radius [Earth radii], shape ``(K,)``.
        log_gravity_cgs: Log10 surface gravity [cm/s^2], shape ``(K,)``.
        species: Tuple of :class:`MolecularSpecies`. Each species owns
            its own ``log_mmr`` (per-planet, shape ``(K,)``). The
            number and identity of species is configurable via
            :func:`build_exojax_engines`.
        bulk: Optional :class:`BulkGasResidual` (implicit residual gas).

    Components (each owns its own per-planet PyTree leaves where
    applicable):
        tp_profile: T-P profile component (e.g. ``PowerLawTPProfile``).
        absorption: Absorption orchestrator (e.g. ``Absorption``).
        scattering: Scattering component (e.g. ``RayleighScattering``,
            ``NullScattering``).
        clouds: Cloud component (e.g. ``GrayCloud``, ``NoCloud``).
        surface: Surface component (e.g. ``WavelengthDependentSurface``).

    Shared / configuration attributes:
        rt_engine: ExoJAX ``ArtReflectPure`` instance.
        nu_grid: Wavenumber grid [cm^-1].
        n_nu: Length of ``nu_grid`` (static for JIT).
    """

    # Top-level per-planet state
    Rp_Rearth: Array
    log_gravity_cgs: Array

    # Atmospheric composition
    species: tuple[MolecularSpecies, ...]
    bulk: BulkGasResidual | None

    # Components
    tp_profile: AbstractTPProfile
    absorption: AbstractAbsorption
    scattering: AbstractScattering
    clouds: AbstractClouds
    surface: AbstractSurface

    # Static / shared
    rt_engine: Any = eqx.field(static=True)
    nu_grid: Array
    n_nu: int = eqx.field(static=True)

    @classmethod
    def from_default_setup(
        cls,
        *,
        Rp_Rearth: Array,
        log_mmrs: dict[str, Array],
        T_eq_K: Array,
        T_alpha: Array,
        log_surface_albedo: Array,
        log_gravity_cgs: Array,
        log_cloud_pressure_bar: Array | None = None,
        log_cloud_opt_depth: Array | None = None,
        surface_albedo_spectrum: Array | None = None,
        molecules: tuple[str, ...] | None = None,
        bulk_gas: str | None = "N2",
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 1000.0,
        n_wavenumbers: int = 2000,
        n_layers: int = 100,
        pressure_top_bar: float = 1.0e-5,
        pressure_btm_bar: float = 1.0e0,
        databases_dir: str | None = None,
        crit: float = 0.0,
    ) -> ExoJaxAtmosphere:
        """One-shot convenience: build engines + default components in one call.

        Defaults to Earth-like physics: ``PowerLawTPProfile``,
        ``Absorption``, ``RayleighScattering`` with N2 bulk,
        ``GrayCloud`` (Earth water clouds), and a flat
        ``WavelengthDependentSurface``.

        Args:
            Rp_Rearth: Planet radii [R_earth], shape ``(K,)``.
            log_mmrs: Dict mapping molecule name to per-planet log10
                mass-mixing ratio, shape ``(K,)`` each. The dict's
                molecules determine which species are built.
            T_eq_K: Per-planet equatorial T [K], ``(K,)``.
            T_alpha: Per-planet T-P power-law exponent, ``(K,)``.
            log_surface_albedo: Per-planet surface scaling, ``(K,)``.
            log_gravity_cgs: Per-planet log10 gravity, ``(K,)``.
            log_cloud_pressure_bar: Per-planet cloud-deck pressure, ``(K,)``.
            log_cloud_opt_depth: Per-planet cloud optical depth, ``(K,)``.
            surface_albedo_spectrum: ``(n_nu,)`` spectral profile.
                Defaults to flat ones.
            molecules: Override molecule list (default: derived from
                ``log_mmrs.keys()``).
            bulk_gas: Implicit residual gas (default ``"N2"``).
            wavelength_min_nm: See :func:`build_exojax_engines`.
            wavelength_max_nm: See :func:`build_exojax_engines`.
            n_wavenumbers: See :func:`build_exojax_engines`.
            n_layers: See :func:`build_exojax_engines`.
            pressure_top_bar: See :func:`build_exojax_engines`.
            pressure_btm_bar: See :func:`build_exojax_engines`.
            databases_dir: See :func:`build_exojax_engines`.
            crit: See :func:`build_exojax_engines`.
        """
        if molecules is None:
            molecules = tuple(log_mmrs.keys())
        engines = build_exojax_engines(
            molecules=molecules,
            bulk_gas=bulk_gas,
            wavelength_min_nm=wavelength_min_nm,
            wavelength_max_nm=wavelength_max_nm,
            n_wavenumbers=n_wavenumbers,
            n_layers=n_layers,
            pressure_top_bar=pressure_top_bar,
            pressure_btm_bar=pressure_btm_bar,
            databases_dir=databases_dir,
            crit=crit,
        )
        K = Rp_Rearth.shape[0]
        if log_cloud_pressure_bar is None:
            log_cloud_pressure_bar = jnp.full((K,), jnp.log10(0.5))
        if log_cloud_opt_depth is None:
            log_cloud_opt_depth = jnp.full((K,), jnp.log10(5.0))
        if surface_albedo_spectrum is None:
            surface_albedo_spectrum = jnp.ones(engines["n_nu"])

        species = _assemble_species(log_mmrs, engines["species_prebuilt"])
        bulk = _assemble_bulk(engines["bulk_prebuilt"])

        return cls(
            Rp_Rearth=Rp_Rearth,
            log_gravity_cgs=log_gravity_cgs,
            species=species,
            bulk=bulk,
            tp_profile=PowerLawTPProfile(T_eq_K=T_eq_K, T_alpha=T_alpha),
            absorption=Absorption(),
            scattering=RayleighScattering(),
            clouds=GrayCloud(
                log_pressure_bar=log_cloud_pressure_bar,
                log_opt_depth=log_cloud_opt_depth,
            ),
            surface=WavelengthDependentSurface(
                log_albedo=log_surface_albedo,
                spectrum=surface_albedo_spectrum,
            ),
            rt_engine=engines["rt_engine"],
            nu_grid=engines["nu_grid"],
            n_nu=engines["n_nu"],
        )

    @classmethod
    def from_default_setup_cached(
        cls,
        *,
        Rp_Rearth: Array,
        log_mmrs: dict[str, Array | AbstractMmrProfile],
        T_eq_K: Array,
        T_alpha: Array,
        log_surface_albedo: Array,
        log_gravity_cgs: Array,
        log_cloud_pressure_bar: Array | None = None,
        log_cloud_opt_depth: Array | None = None,
        surface_albedo_spectrum: Array | None = None,
        molecules: tuple[str, ...] | None = None,
        bulk_gas: str | None = "N2",
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 1000.0,
        n_wavenumbers: int = 2000,
        n_layers: int = 100,
        pressure_top_bar: float = 1.0e-5,
        pressure_btm_bar: float = 1.0e0,
        databases_dir: str | None = None,
        crit: float = 0.0,
        cache_dir: str | Path | None = None,
    ) -> PrecomputedReflectivity:
        """Cached one-shot factory: returns a fast :class:`PrecomputedReflectivity`.

        Hashes every input and looks up a cached spectrum on disk. On
        cache hit, returns the cached spectrum in ~10 ms. On cache
        miss, builds the full :class:`ExoJaxAtmosphere` via
        :meth:`from_default_setup`, runs the 2-stream RT once,
        precomputes the reflectivity, saves to disk, and returns the
        :class:`PrecomputedReflectivity`.

        Use this when the atmosphere parameters are fixed across many
        evaluations (coronagraphoto sims, ETC studies). For HMC
        retrievals where parameters vary, use :meth:`from_default_setup`
        directly.

        Args:
            cache_dir: Override the cache directory. Defaults to
                ``~/.cache/skyscapes/atmospheres/``.
            Rp_Rearth: See :meth:`from_default_setup`.
            log_mmrs: See :meth:`from_default_setup`.
            T_eq_K: See :meth:`from_default_setup`.
            T_alpha: See :meth:`from_default_setup`.
            log_surface_albedo: See :meth:`from_default_setup`.
            log_gravity_cgs: See :meth:`from_default_setup`.
            log_cloud_pressure_bar: See :meth:`from_default_setup`.
            log_cloud_opt_depth: See :meth:`from_default_setup`.
            surface_albedo_spectrum: See :meth:`from_default_setup`.
            molecules: See :meth:`from_default_setup`.
            bulk_gas: See :meth:`from_default_setup`.
            wavelength_min_nm: See :meth:`from_default_setup`.
            wavelength_max_nm: See :meth:`from_default_setup`.
            n_wavenumbers: See :meth:`from_default_setup`.
            n_layers: See :meth:`from_default_setup`.
            pressure_top_bar: See :meth:`from_default_setup`.
            pressure_btm_bar: See :meth:`from_default_setup`.
            databases_dir: See :meth:`from_default_setup`.
            crit: See :meth:`from_default_setup`.

        Returns:
            A :class:`PrecomputedReflectivity` with no RT cost at
            evaluation time.
        """
        kwargs = {
            "Rp_Rearth": Rp_Rearth,
            "log_mmrs": log_mmrs,
            "T_eq_K": T_eq_K,
            "T_alpha": T_alpha,
            "log_surface_albedo": log_surface_albedo,
            "log_gravity_cgs": log_gravity_cgs,
            "log_cloud_pressure_bar": log_cloud_pressure_bar,
            "log_cloud_opt_depth": log_cloud_opt_depth,
            "surface_albedo_spectrum": surface_albedo_spectrum,
            "molecules": molecules,
            "bulk_gas": bulk_gas,
            "wavelength_min_nm": wavelength_min_nm,
            "wavelength_max_nm": wavelength_max_nm,
            "n_wavenumbers": n_wavenumbers,
            "n_layers": n_layers,
            "pressure_top_bar": pressure_top_bar,
            "pressure_btm_bar": pressure_btm_bar,
            "crit": crit,
        }
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "skyscapes" / "atmospheres"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_cache_key(**kwargs)}.npz"

        if cache_path.exists():
            return PrecomputedReflectivity.load(cache_path)

        atm = cls.from_default_setup(databases_dir=databases_dir, **kwargs)
        cached = PrecomputedReflectivity.from_atmosphere(atm)
        cached.save(cache_path)
        return cached

    def _reflectivity_one_planet(self, k: int) -> Array:
        """Plane-parallel reflectivity for the k-th planet, shape ``(n_nu,)``.

        Indexes every K-shaped leaf at position ``k`` (cheap; JIT
        traces this as plain array indexing). With per-species mmr
        profiles the previous "vmap over an extracted log_mmrs array"
        pattern broke because different profile types have different
        K-shaped fields -- a Python loop over K avoids that issue
        without losing JIT efficiency at K=1 (the common case).
        """
        pressure = self.rt_engine.pressure
        n_layers = pressure.shape[0]
        log_gravity_cgs_scalar = self.log_gravity_cgs[k]
        gravity = 10.0**log_gravity_cgs_scalar

        Tarr = self.tp_profile.compute_Tarr(
            self.rt_engine,
            self.tp_profile.T_eq_K[k],
            self.tp_profile.T_alpha[k],
        )

        # Index each species' profile at planet k. Tree-map replaces
        # K-shaped array leaves with their k-th slice while leaving
        # static fields (name, molmass, opa) untouched.
        def _index_at_k(tree):
            return jax.tree.map(
                lambda x: x[k] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
                tree,
            )

        species_one = tuple(
            eqx.tree_at(lambda s: s.profile, s, replace=_index_at_k(s.profile))
            for s in self.species
        )

        abs_c = self.absorption.compute(
            species_one, Tarr, pressure, gravity, self.rt_engine
        )
        scat_c = self.scattering.compute(
            species_one, self.bulk, gravity, self.rt_engine, n_layers, self.n_nu
        )
        cloud_c = self.clouds.compute(
            self.clouds.log_pressure_bar[k],
            self.clouds.log_opt_depth[k],
            pressure,
            self.n_nu,
        )

        dtau_total = abs_c.dtau_total + scat_c.dtau_total + cloud_c.dtau_total
        dtau_scatter = abs_c.dtau_scatter + scat_c.dtau_scatter + cloud_c.dtau_scatter
        g_num = abs_c.g_weighted_num + scat_c.g_weighted_num + cloud_c.g_weighted_num

        ssa = jnp.where(dtau_total > 0, dtau_scatter / dtau_total, 0.0)
        g_asym = jnp.where(dtau_scatter > 0, g_num / dtau_scatter, 0.0)

        refl_surface = self.surface.compute_refl(self.surface.log_albedo[k], self.n_nu)
        F_unit = jnp.ones(self.n_nu)
        return self.rt_engine.run(dtau_total, ssa, g_asym, refl_surface, F_unit)

    def _reflectivity_all_planets(self) -> Array:
        """Plane-parallel reflectivity for every planet, shape ``(K, n_nu)``.

        Loops over K planets in Python. JIT unrolls the loop; for the
        common K=1 case this is a single iteration with no overhead.
        For HMC retrievals (K=1 per sample), the JIT cache is reused
        across MCMC steps.
        """
        K = int(self.Rp_Rearth.shape[0])
        return jnp.stack([self._reflectivity_one_planet(k) for k in range(K)], axis=0)

    def reflected_spectrum(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelength_nm: Array,
    ) -> Array:
        """Per-planet, per-time geometric-albedo contrast at one wavelength.

        Args:
            phase_angle_rad: Star-planet-observer phase angle, ``(K, T)``.
            dist_AU: Star-planet distance [AU], ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm].

        Returns:
            Contrast = ``A_g(lambda) * Lambert_phase(beta) * (Rp/d)^2``,
            shape ``(K, T)``. The 2-stream RT produces the plane-
            parallel (spherical) reflectivity; we convert to geometric
            albedo via the Lambertian-sphere factor 2/3 (Seager 2010,
            eq 3.36) so the output is a geometric-albedo contrast --
            the same convention as :class:`LambertianAtmosphere`.
        """
        R_per_planet = self._reflectivity_all_planets()
        target_nu_cm_inv = nm_to_wavenumber_cm(wavelength_nm)

        def interp_one(spectrum):
            return interpax.interp1d(
                target_nu_cm_inv,
                self.nu_grid,
                spectrum,
                method="cubic",
                extrap=True,
            )

        R_at_wl = jax.vmap(interp_one)(R_per_planet)

        Rp_AU = (self.Rp_Rearth * Rearth2AU)[:, None]
        Ag = spherical_to_geometric_albedo(R_at_wl)[:, None]
        phase = _lambert_phase(phase_angle_rad)
        return Ag * phase * (Rp_AU / dist_AU) ** 2

    def reflected_spectrum_cube(
        self,
        phase_angle_rad: Array,
        dist_AU: Array,
        wavelengths_nm: Array,
    ) -> Array:
        """Per-planet, per-time contrast across many wavelengths.

        Returns ``(W, K, T)`` geometric-albedo contrast cube. Computes
        the underlying 2-stream RT exactly once per planet rather
        than once per wavelength; applies the spherical-to-geometric
        Lambertian-sphere conversion (Seager 2010, eq 3.36) the same
        way as :meth:`reflected_spectrum`.
        """
        R_per_planet = self._reflectivity_all_planets()
        target_nu_cm_inv = nm_to_wavenumber_cm(wavelengths_nm)

        def interp_one_planet(spectrum):
            return interpax.interp1d(
                target_nu_cm_inv,
                self.nu_grid,
                spectrum,
                method="cubic",
                extrap=True,
            )

        R_at_wls = spherical_to_geometric_albedo(
            jax.vmap(interp_one_planet)(R_per_planet)
        )  # (K, W)

        Rp_AU = (self.Rp_Rearth * Rearth2AU)[:, None]
        phase = _lambert_phase(phase_angle_rad)
        geom = phase * (Rp_AU / dist_AU) ** 2

        return jnp.einsum("kw,kt->wkt", R_at_wls, geom)

    def __repr__(self) -> str:
        """Human-readable summary of components + species + per-planet state."""
        K = int(self.Rp_Rearth.shape[0])
        n_layers = int(self.rt_engine.pressure.shape[0])
        wl_min_nm = float(1.0e7 / self.nu_grid.max())
        wl_max_nm = float(1.0e7 / self.nu_grid.min())
        bulk_label = self.bulk.name if self.bulk is not None else "None"
        species_label = ", ".join(s.name for s in self.species)

        lines = [
            f"ExoJaxAtmosphere(K={K})",
            (
                f"  Wavelength: {wl_min_nm:.0f}-{wl_max_nm:.0f} nm "
                f"(n_nu={self.n_nu}, n_layers={n_layers})"
            ),
            f"  Composition: [{species_label}] + bulk={bulk_label}",
            "  Components:",
            indent(f"tp: {type(self.tp_profile).__name__}", prefix="    "),
            indent(
                f"absorption: {type(self.absorption).__name__}",
                prefix="    ",
            ),
            indent(
                f"scattering: {type(self.scattering).__name__}",
                prefix="    ",
            ),
            indent(f"clouds: {type(self.clouds).__name__}", prefix="    "),
            indent(f"surface: {type(self.surface).__name__}", prefix="    "),
        ]

        max_show = 3
        pressure = self.rt_engine.pressure
        # Display MMR at the bottom-of-atmosphere layer (typically 1 bar)
        # as the "headline" value -- this matches the surface/tropospheric
        # value most readers expect when they see "Earth H2O ~ 3e-3".
        botm_layer_idx = int(jnp.argmax(pressure))
        for k in range(min(K, max_show)):
            # Index each species' profile at planet k, then evaluate
            # at the bottom-of-atmosphere pressure for the display.
            mmrs_k = []
            profile_labels = []

            def _index_k(x, k=k):
                return x[k] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x

            for s in self.species:
                profile_k = jax.tree.map(_index_k, s.profile)
                mmr_at_botm = float(profile_k.evaluate(pressure)[botm_layer_idx])
                mmrs_k.append(mmr_at_botm)
                profile_labels.append(type(s.profile).__name__)
            mmr_bulk_k = max(0.0, 1.0 - sum(mmrs_k))
            inv_m_mean = sum(
                mmr / s.molmass for s, mmr in zip(self.species, mmrs_k, strict=True)
            )
            if self.bulk is not None:
                inv_m_mean += mmr_bulk_k / self.bulk.molmass
            m_mean = 1.0 / inv_m_mean if inv_m_mean > 0 else 28.0

            rp = float(self.Rp_Rearth[k])
            g = 10.0 ** float(self.log_gravity_cgs[k])
            t_eq = float(self.tp_profile.T_eq_K[k])
            alpha = float(self.tp_profile.T_alpha[k])
            ag = 10.0 ** float(self.surface.log_albedo[k])
            p_cl = 10.0 ** float(self.clouds.log_pressure_bar[k])
            tau_cl = 10.0 ** float(self.clouds.log_opt_depth[k])

            lines += [
                f"  Planet {k}:",
                f"    Rp = {rp:.3f} R_earth, g = {g:.0f} cm/s^2",
                f"    T(P=1bar) = {t_eq:.1f} K, alpha = {alpha:.3f}",
                f"    Cloud: P = {p_cl:.2g} bar, tau = {tau_cl:.2g}",
                f"    Surface albedo (scalar) = {ag:.3f}",
                (
                    f"    Mixing ratios at bottom layer "
                    f"(MMR / VMR, M_mean = {m_mean:.2f} g/mol):"
                ),
            ]
            for s, mmr, lbl in zip(self.species, mmrs_k, profile_labels, strict=True):
                vmr = mmr * m_mean / s.molmass
                lines.append(f"      {s.name:<5s} {mmr:.3e} / {vmr:.3e}  [{lbl}]")
            if self.bulk is not None:
                vmr_bulk = mmr_bulk_k * m_mean / self.bulk.molmass
                lines.append(
                    f"      {self.bulk.name}*  {mmr_bulk_k:.3e} / "
                    f"{vmr_bulk:.3e}  (* implicit residual)"
                )

        if K > max_show:
            lines.append(f"  ... and {K - max_show} more planets")

        return "\n".join(lines)

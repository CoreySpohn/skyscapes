"""CIA file, interpolation and hydrostatic optical-depth contracts."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.physical_model.exojax import (
    ConstantMmr,
    ExoJaxEmissionModel,
    MolecularSpecies,
    StratosphericPeakMmr,
    build_exojax_emission_engines,
)
from skyscapes.physical_model.exojax.cia import load_cia

FILES = {
    "N2-N2": "N2-N2_2021.cia",
    "O2-O2": "O2-O2_2024.cia",
    "O2-N2": "O2-N2_2024.cia",
    "CO2-CO2": "CO2-CO2_2024.cia",
    "N2-H2O": "N2-H2O_2018.cia",
}
CACHE = Path.home() / ".cache" / "skyscapes" / "cia"


def _write_cia(path, pair, blocks):
    with path.open("w") as stream:
        for temperature, nu, coefficients in blocks:
            stream.write(f"{pair:>20} {nu[0]} {nu[-1]} {len(nu)} {temperature} 0 0\n")
            for x, y in zip(nu, coefficients, strict=True):
                stream.write(f"{x} {y} 0\n")  # optional uncertainty column


def _model(engines, **kwargs):
    return ExoJaxEmissionModel.from_engines(
        engines=engines,
        species=kwargs.pop("species", ()),
        temperature_coefficients=jnp.array([[250.0]]),
        log_gravity_cgs=jnp.array([3.0]),
        surface_temperature_K=jnp.array([300.0]),
        **kwargs,
    )


def test_empty_cia_is_bitwise_identical(tmp_path):
    """Explicitly disabling CIA preserves the existing nonisothermal output."""
    kwargs = dict(molecules=(), n_wavenumbers=20, n_layers=3)
    original = _model(build_exojax_emission_engines(**kwargs)).emergent_flux()
    explicit = _model(
        build_exojax_emission_engines(**kwargs, cia_pairs=(), cia_dir=tmp_path)
    ).emergent_flux()
    np.testing.assert_array_equal(original, explicit)


def test_missing_cia_names_expected_path(tmp_path):
    """Missing CIA fails locally before any line-list loading can download."""
    with pytest.raises(FileNotFoundError) as exc:
        build_exojax_emission_engines(cia_pairs=("O2-O2",), cia_dir=tmp_path)
    assert str(tmp_path / FILES["O2-O2"]) in str(exc.value)
    assert "https://hitran.org/cia/" in str(exc.value)


def test_cia_variable_grids_temperatures_and_spectral_gaps(tmp_path):
    """Disjoint bands stay zero between them; temperature rows sort correctly."""
    _write_cia(
        tmp_path / FILES["N2-N2"],
        "N2-N2",
        [
            (300, [100, 200, 300], [2e-44, 4e-44, 6e-44]),
            (200, [100, 300], [1e-44, 3e-44]),
            (200, [500, 600], [8e-44, 8e-44]),
        ],
    )
    cia = load_cia("N2-N2", tmp_path, np.array([50, 100, 200, 300, 400, 550, 700]))
    actual = cia.coefficient_matrix(jnp.array([250.0]))[0]
    np.testing.assert_allclose(
        actual, [0, 1.5e-44, 3e-44, 4.5e-44, 0, 8e-44, 0], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda c: c.coefficient_matrix(jnp.array([250.0])))(cia)[0],
        actual,
        rtol=1e-12,
        atol=0,
    )


@pytest.fixture
def constant_engines(tmp_path):
    """A uniform cm^5 continuum with two temperature samples."""
    _write_cia(
        tmp_path / FILES["O2-N2"],
        "O2-N2",
        [
            (200, [500, 3000], [1e-45, 1e-45]),
            (300, [500, 3000], [3e-45, 3e-45]),
        ],
    )
    return build_exojax_emission_engines(
        molecules=(),
        n_wavenumbers=20,
        n_layers=3,
        cia_pairs=("O2-N2",),
        cia_dir=tmp_path,
    )


def test_cia_uses_profile_vmr_and_hydrostatic_column(constant_engines):
    """Independent cgs normalization catches MMR/VMR or pressure-unit errors."""
    species = (
        MolecularSpecies(
            name="O2",
            molmass=32.0,
            opa=None,
            rayleigh_xs=jnp.zeros(20),
            profile=ConstantMmr(jnp.array([np.log10(0.2 * 32.0 / 28.97)])),
        ),
    )
    model = _model(constant_engines, species=species, cia_vmr={"N2": 0.78})
    actual = model._cia_optical_depth(0)
    pressure = np.asarray(model.rt_engine.pressure)
    width = np.asarray(model.rt_engine.dParr)
    n = pressure * 1e6 / (1.380649e-16 * 250)
    dz = 1.380649e-16 * 250 / (28.97 * 1.66053906660e-24 * 1e3) * width / pressure
    expected = 2e-45 * (0.2 * n) * (0.78 * n) * dz
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected[:, None], actual.shape), rtol=1e-7
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda m: m.emergent_flux())(model),
        model.emergent_flux(),
        rtol=1e-12,
    )

    def flux(vmr):
        changed = eqx.tree_at(lambda m: m.cia_vmr["N2"], model, vmr)
        return jnp.mean(changed.emergent_flux())

    assert np.isfinite(jax.grad(flux)(jnp.array(0.78)))
    assert jax.grad(flux)(jnp.array(0.78)) < 0
    assert np.isfinite(jax.grad(flux)(jnp.array(0.0)))


def test_missing_partner_is_explicit(constant_engines):
    """An unspecified filler must not silently remove an enabled continuum."""
    with pytest.raises(ValueError, match="cia_vmr"):
        _model(constant_engines, cia_vmr={"N2": 0.78})


@pytest.mark.parametrize("pair", FILES)
def test_cached_cia_matches_tabulated_point(pair):
    """A direct file sample is reproduced at its temperature and frequency."""
    path = CACHE / FILES[pair]
    if not path.exists():
        pytest.skip(f"data-absent: {path}")
    with path.open() as stream:
        header = stream.readline().split()
        rows = np.array(
            [
                list(map(float, stream.readline().split()[:2]))
                for _ in range(int(header[3]))
            ]
        )
    index = np.argmax(rows[:, 1])
    nu, expected = rows[index]
    cia = load_cia(pair, CACHE, np.array([nu]))
    actual = cia.coefficient_matrix(jnp.array([float(header[4])]))[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_cia_profiles_follow_planet_and_layer(constant_engines):
    """CIA uses each planet's varying profile, even with a conflicting constant."""
    profile = StratosphericPeakMmr(
        log_peak_mmr=jnp.array([-1.0, -2.0]),
        log_peak_pressure_bar=jnp.array([-1.0, -1.0]),
        log_sigma_decades=jnp.array([0.5, 0.5]),
    )
    species = (
        MolecularSpecies(
            profile=profile,
            name="O2",
            molmass=32.0,
            opa=None,
            rayleigh_xs=jnp.zeros(20),
        ),
    )
    model = ExoJaxEmissionModel.from_engines(
        engines=constant_engines,
        species=species,
        temperature_coefficients=jnp.array([[250.0], [250.0]]),
        log_gravity_cgs=jnp.array([3.0, 3.0]),
        cia_vmr={"N2": 0.78, "O2": 0.0},
    )
    tau0 = model._cia_optical_depth(0)
    tau1 = model._cia_optical_depth(1)
    np.testing.assert_allclose(tau1, 0.1 * tau0, rtol=1e-12)
    pressure = np.asarray(model.rt_engine.pressure)
    reduced = np.asarray(tau0[:, 0]) / (pressure * model.rt_engine.dParr)
    shape = np.exp(-0.5 * ((np.log10(pressure) + 1) / 0.5) ** 2)
    np.testing.assert_allclose(reduced / reduced[-1], shape / shape[-1], rtol=1e-12)


def test_cia_constructs_under_jit_and_temperature_gradient(constant_engines):
    """Temperature interpolation remains live during traced construction."""

    def construct(temperature):
        model = ExoJaxEmissionModel.from_engines(
            engines=constant_engines,
            species=(),
            temperature_coefficients=temperature.reshape(1, 1),
            log_gravity_cgs=jnp.array([3.0]),
            surface_temperature_K=jnp.array([300.0]),
            cia_vmr={"N2": 0.78, "O2": 0.21},
        )
        return jnp.mean(model.emergent_flux())

    temperature = jnp.array(250.0)
    np.testing.assert_allclose(
        jax.jit(construct)(temperature), construct(temperature), rtol=1e-12
    )
    step = 1e-2
    expected = (construct(temperature + step) - construct(temperature - step)) / (
        2 * step
    )
    np.testing.assert_allclose(jax.grad(construct)(temperature), expected, rtol=1e-6)


def test_adjacent_cia_bands_with_overlapping_edges_keep_both(tmp_path):
    """A small edge overlap must not erase an adjacent band's temperature row."""
    _write_cia(
        tmp_path / FILES["O2-O2"],
        "O2-O2",
        [
            (200, [100, 200, 300], [1e-44, 1e-44, 1e-44]),
            (200, [290, 400, 500], [2e-44, 2e-44, 2e-44]),
        ],
    )
    cia = load_cia("O2-O2", tmp_path, np.array([150, 295, 450]))
    np.testing.assert_allclose(
        cia.coefficient_matrix(jnp.array([200.0]))[0],
        [1e-44, 3e-44, 2e-44],
        rtol=1e-12,
        atol=0,
    )


def test_repeated_temperature_only_replaces_covered_frequencies(tmp_path):
    """A later alternate measurement cannot delete an uncovered band wing."""
    _write_cia(
        tmp_path / FILES["O2-O2"],
        "O2-O2",
        [
            (200, [100, 200, 300, 400], [1e-44] * 4),
            (200, [200, 300, 400, 500], [2e-44] * 4),
        ],
    )
    cia = load_cia("O2-O2", tmp_path, np.array([150, 250, 450]))
    np.testing.assert_allclose(
        cia.coefficient_matrix(jnp.array([200.0]))[0],
        [1e-44, 2e-44, 2e-44],
        rtol=1e-12,
        atol=0,
    )

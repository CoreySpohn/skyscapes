"""CIA file, interpolation and hydrostatic optical-depth contracts."""

from pathlib import Path
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testing import precision_pair, ulps

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


@pytest.mark.parametrize(
    "molecules",
    [
        pytest.param((), id="continuum"),
        pytest.param(
            ("O2",),
            marks=pytest.mark.float64_only(reason="PreMODIT line opacity"),
            id="with-lines",
        ),
    ],
)
def test_missing_cia_names_expected_path(tmp_path, molecules):
    """Missing CIA fails locally before any line-list loading can download."""
    with (
        patch(
            "skyscapes.physical_model.exojax.components.species.MdbHitran",
            side_effect=AssertionError("opened HITRAN before checking CIA"),
        ) as reader,
        pytest.raises(FileNotFoundError) as exc,
    ):
        build_exojax_emission_engines(
            molecules=molecules,
            cia_pairs=("O2-O2",),
            cia_dir=tmp_path,
            databases_dir=str(tmp_path),
            n_wavenumbers=400,
        )
    reader.assert_not_called()
    assert str(tmp_path / FILES["O2-O2"]) in str(exc.value)
    assert "https://hitran.org/cia/" in str(exc.value)


def test_cia_variable_grids_temperatures_and_spectral_gaps(tmp_path, precision):
    """Disjoint bands stay zero and the float64 coefficient survives narrowing.

    Floating-point bound: one conversion, two linear interpolations (four
    operations each) and one band sum, in nm^5 per molecule pair.
    """
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
    temperatures = jnp.array([250.0])
    actual = cia.coefficient_matrix_nm5(temperatures)[0]
    rtol = float(ulps(10, 1.0))
    if precision == np.float64:
        precision_pair(
            lambda table, t: table.coefficient_matrix_nm5(t),
            (cia, temperatures),
            bound=float(ulps(10, np.float32(1.0))),
            scale=0.0,
        )
    np.testing.assert_allclose(
        actual, [0, 1.5e-9, 3e-9, 4.5e-9, 0, 8e-9, 0], rtol=rtol, atol=0
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda c: c.coefficient_matrix_nm5(jnp.array([250.0])))(cia)[0],
        actual,
        rtol=rtol,
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
    """Independent cgs normalization catches MMR/VMR or pressure-unit errors.

    The reference allows 3e-8 for the ExoJAX atomic mass and 20 roundings
    across density, scale height, pressure widths, VMRs and interpolation.
    The flux has three layers and four rays. Its Planck exponent is below
    18, allowing 4 * 18 roundings for the exponent and 20 for the source,
    transmissions, and ray sums when comparing eager and fused evaluation.
    """
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
        actual,
        np.broadcast_to(expected[:, None], actual.shape),
        rtol=3e-8 + float(ulps(20, 1.0)),
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda m: m.emergent_flux())(model),
        model.emergent_flux(),
        rtol=float(ulps(4 * 18 + 20, 1.0)),
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
@pytest.mark.slow
@pytest.mark.float64_only(reason="float64 HITRAN coefficient reference")
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
    actual = cia.coefficient_matrix_nm5(jnp.array([float(header[4])]))[0, 0]
    np.testing.assert_allclose(actual, expected * 1e35, rtol=1e-6, atol=0)


def test_cia_profiles_follow_planet_and_layer(constant_engines):
    """CIA uses each planet's varying profile, even with a conflicting constant.

    Floating-point bound: the optical-depth ratio has at most 20 products,
    divisions and profile operations. The Gaussian shape's log pressure
    contributes up to 2 * |log10(P)+1| / sigma^2 conditioned roundings.
    """
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
    np.testing.assert_allclose(tau1, 0.1 * tau0, rtol=float(ulps(20, 1.0)))
    pressure = np.asarray(model.rt_engine.pressure)
    reduced = np.asarray(tau0[:, 0]) / (pressure * model.rt_engine.dParr)
    shape = np.exp(-0.5 * ((np.log10(pressure) + 1) / 0.5) ** 2)
    condition = 2 * np.max(np.abs(np.log10(pressure) + 1)) / 0.5**2
    np.testing.assert_allclose(
        reduced / reduced[-1],
        shape / shape[-1],
        rtol=float(ulps(20 + int(np.ceil(condition)), 1.0)),
    )


@pytest.mark.float64_only(reason="finite-difference derivative")
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
    """Both bands contribute in the overlap.

    Three roundings cover coefficient conversion, storage and addition.
    """
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
        cia.coefficient_matrix_nm5(jnp.array([200.0]))[0],
        [1e-9, 3e-9, 2e-9],
        rtol=float(ulps(3, 1.0)),
        atol=0,
    )


def test_repeated_temperature_only_replaces_covered_frequencies(tmp_path):
    """A later alternate measurement cannot delete an uncovered band wing.

    Two roundings cover the conversion to nm^5 and table storage.
    """
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
        cia.coefficient_matrix_nm5(jnp.array([200.0]))[0],
        [1e-9, 2e-9, 2e-9],
        rtol=float(ulps(2, 1.0)),
        atol=0,
    )


def test_emission_model_refuses_a_shared_layer_array(constant_engines):
    """Fails if a layer-length array is accepted where a per-planet one is required.

    Truth: a construction check with no tolerance. With two planets and three
    layers, a length-3 profile would be sliced as if its layers were planets;
    scalar-per-planet and (planets, n) profiles are accepted.
    """
    kwargs = dict(
        engines=constant_engines,
        temperature_coefficients=jnp.array([[250.0], [250.0]]),
        log_gravity_cgs=jnp.array([3.0, 3.0]),
        cia_vmr={"N2": 0.78, "O2": 0.0},
    )

    def species(log_mmr):
        return (
            MolecularSpecies(
                profile=ConstantMmr(log_mmr),
                name="O2",
                molmass=32.0,
                opa=None,
                rayleigh_xs=jnp.zeros(20),
            ),
        )

    for ok in (jnp.array(-1.0), jnp.full((2,), -1.0), jnp.full((2, 5), -1.0)):
        ExoJaxEmissionModel.from_engines(species=species(ok), **kwargs)
    with pytest.raises(ValueError, match="layer count"):
        ExoJaxEmissionModel.from_engines(
            species=species(jnp.full((3,), -1.0)), **kwargs
        )

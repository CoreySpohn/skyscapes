"""Tests for :mod:`skyscapes.atmosphere.exojax.o3_chappuis`.

Avoids network access by writing a small synthetic PSG-format file to
a tmp path and pointing the adapter at it. The real Serdyuchenko table
is exercised by the slow ExoJAX integration test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from skyscapes.atmosphere.exojax.o3_chappuis import (
    O3_MOLMASS,
    O3ChappuisOpacity,
    _load_psg_xs_table,
)


def _write_fake_psg_file(path):
    """Write a minimal PSG-format cross-section file.

    Three temperatures (200, 250, 300 K), five wavelengths spanning a
    fake Chappuis-like peak around 600 nm. The values are made up but
    exercise the parser and interpolator without a network fetch.
    """
    body = (
        "#MOLECULE:O3\n"
        "#REF:fake for tests\n"
        "#TYPE:3   ! 3=Cross_section[cm2/molecule]\n"
        "#TEMP: 200 250 300   ! Temperature(s) [K]\n"
        "#POINTS:5\n"
        "    0.400  1.000e-25  1.100e-25  1.200e-25\n"
        "    0.500  3.000e-22  3.100e-22  3.200e-22\n"
        "    0.600  5.000e-22  5.500e-22  6.000e-22\n"
        "    0.700  2.000e-22  2.200e-22  2.400e-22\n"
        "    0.800  1.000e-25  1.100e-25  1.200e-25\n"
    )
    with open(path, "w") as f:
        f.write(body)


def test_load_psg_xs_table_parses_header_and_data(tmp_path):
    """Parser pulls T-grid from #TEMP: line and a (N, T) sigma matrix."""
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)

    wavelength_um, T_grid, sigma = _load_psg_xs_table(path)

    assert wavelength_um.shape == (5,)
    assert np.allclose(T_grid, [200.0, 250.0, 300.0])
    assert sigma.shape == (5, 3)
    # Sanity: the 600-nm row has the largest peak.
    assert sigma[2, 0] == 5.0e-22


def test_xsmatrix_shape_and_dtype(tmp_path):
    """Xsmatrix returns shape (n_layers, n_nu) of float dtype."""
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)

    # nu_grid covering 400--800 nm = 12500--25000 cm^-1.
    n_nu = 50
    nu_grid = jnp.linspace(12500.0, 25000.0, n_nu)
    adapter = O3ChappuisOpacity(nu_grid, xs_table_path=path)

    Tarr = jnp.array([220.0, 250.0, 280.0])
    pressure = jnp.array([0.1, 0.5, 1.0])  # unused, but contract requires
    xs = adapter.xsmatrix(Tarr, pressure)

    assert xs.shape == (3, n_nu)
    assert xs.dtype == jnp.result_type(adapter._sigma_at_nu, Tarr)
    assert bool(jnp.all(jnp.isfinite(xs)))
    assert bool(jnp.all(xs >= 0.0))


def test_xsmatrix_peaks_in_chappuis_range(tmp_path):
    """The largest cross-section column sits in the Chappuis band (~600 nm).

    With the synthetic table peaking at 0.6 um, the corresponding column
    in nu-space should be the brightest.
    """
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)

    nu_grid = jnp.linspace(12500.0, 25000.0, 200)  # 400--800 nm
    adapter = O3ChappuisOpacity(nu_grid, xs_table_path=path)

    Tarr = jnp.array([250.0])
    pressure = jnp.array([1.0])
    xs = adapter.xsmatrix(Tarr, pressure)  # (1, n_nu)

    nu_peak = float(nu_grid[int(jnp.argmax(xs[0]))])
    wavelength_peak_nm = 1e7 / nu_peak
    # Should fall close to 600 nm given the synthetic table.
    assert 550.0 <= wavelength_peak_nm <= 650.0


def test_xsmatrix_T_clamping(tmp_path):
    """Temperatures outside the table clamp to the nearest endpoint.

    A T below the table's minimum yields the same cross-section as
    the table's minimum-temperature row; same for T above the max.
    """
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)

    nu_grid = jnp.linspace(12500.0, 25000.0, 100)
    adapter = O3ChappuisOpacity(nu_grid, xs_table_path=path)
    pressure = jnp.array([1.0])

    xs_cold = adapter.xsmatrix(jnp.array([100.0]), pressure)
    xs_at_min = adapter.xsmatrix(jnp.array([200.0]), pressure)
    assert jnp.allclose(xs_cold, xs_at_min)

    xs_hot = adapter.xsmatrix(jnp.array([500.0]), pressure)
    xs_at_max = adapter.xsmatrix(jnp.array([300.0]), pressure)
    assert jnp.allclose(xs_hot, xs_at_max)


def test_xsmatrix_jit_stable(tmp_path):
    """Xsmatrix output is identical under JIT vs eager."""
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)

    nu_grid = jnp.linspace(12500.0, 25000.0, 64)
    adapter = O3ChappuisOpacity(nu_grid, xs_table_path=path)
    Tarr = jnp.linspace(210.0, 290.0, 8)
    pressure = jnp.linspace(1.0e-3, 1.0, 8)

    xs_eager = adapter.xsmatrix(Tarr, pressure)
    xs_jit = jax.jit(adapter.xsmatrix)(Tarr, pressure)
    assert jnp.allclose(xs_eager, xs_jit)


def test_molmass_attribute_matches_hitran_value(tmp_path):
    """``molmass`` is the standard HITRAN value for principal O3 isotope."""
    assert O3_MOLMASS == 47.9982
    # After refactor to the generic PsgCrossSectionOpacity base class
    # ``molmass`` is an instance attribute, not class-level.
    path = tmp_path / "fake_o3.txt"
    _write_fake_psg_file(path)
    nu_grid = jnp.linspace(12500.0, 25000.0, 50)
    adapter = O3ChappuisOpacity(nu_grid, xs_table_path=path)
    assert adapter.molmass == O3_MOLMASS

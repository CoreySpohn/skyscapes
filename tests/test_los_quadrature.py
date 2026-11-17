"""Line-of-sight quadrature rules of the parametric disks.

Gauss-Legendre is checked against a high-node reference and its gradients
against central differences. Images are 16 x 16 pixels.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from zodi import DensityDust

from skyscapes.disk import ExovistaParametricDisk, GraterDisk
from skyscapes.disk.grater import two_power_law_density
from tests.conftest import radiance, sky_grid

_GRID = sky_grid(16, 16, 0.015)

_PA_DEG = 30.0


def _ring(n_slices_los: int, **overrides) -> GraterDisk:
    """Narrow ring at 0.7 AU (alpha = 5, h/r = 0.05) with tapered edges."""
    settings = dict(
        amplitude=DensityDust(
            level=1.0, profile_id="grater-1999", calibration_id="unit-profile"
        ),
        sma_AU=jnp.array(0.7),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.array(0.035),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(0.15),
        rmax_AU=jnp.array(2.5),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([1.0, 1.0]),
        n_slices_los=n_slices_los,
        cutoff_width_AU=jnp.array(0.1),
    )
    settings.update(overrides)
    return GraterDisk(**settings)


@eqx.filter_jit
def _image(disk, incl_deg, grid=_GRID):
    return radiance(
        disk,
        jnp.array(500.0),
        jnp.array(0.0),
        incl_deg,
        90.0 - (jnp.array(_PA_DEG)),
        dist_pc=10.0,
        grid=grid,
    )


def _render(disk, incl_deg: float, grid=_GRID) -> np.ndarray:
    return np.asarray(_image(disk, jnp.array(incl_deg), grid))


def _peak_error(image: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(image - reference).max() / reference.max())


@pytest.fixture(scope="module")
def reference():
    """Gauss-Legendre images at 513 nodes, by inclination."""
    disk = _ring(513)
    return {incl: _render(disk, incl) for incl in (45.0, 60.0, 88.0, 90.0)}


@pytest.mark.parametrize(
    ("n_slices_los", "incl", "tolerance"),
    [
        (48, 60.0, 1e-5),
        (96, 88.0, 1e-3),
        (96, 90.0, 1e-3),
    ],
)
def test_rule_converges_to_the_reference(reference, n_slices_los, incl, tolerance):
    """Fails if a rule's nodes or path weights are misplaced.

    Largest pixel error over the image peak, tapered edges. Tolerance basis:
    about ten times the errors measured on this grid (48 nodes at 60 deg:
    4e-8; 96 at 88 deg: 7e-5; 96 at 90 deg: 4.5e-5).
    """
    disk = _ring(n_slices_los)
    error = _peak_error(_render(disk, incl), reference[incl])
    print(f"{n_slices_los} at {incl}: {error:.2e}")
    assert error < tolerance


_FIT_FIELDS = ("sma_AU", "alpha_in", "alpha_out", "ksi0_AU", "gamma", "g_HG_grid")


@pytest.mark.parametrize(
    ("n_slices_los", "incl", "gamma"),
    [
        (32, 45.0, 2.0),
        (32, 88.0, 2.0),
        (32, 90.0, 2.0),
    ],
)
def test_gradients_match_central_differences(n_slices_los, incl, gamma):
    """Fails if a rule's nodes or weights drop a dependence on a parameter.

    A weighted image sum is differentiated with respect to the shape
    parameters, the vertical exponent (which sets the Gauss-Legendre window),
    the scattering
    asymmetry and the inclination, including exactly edge-on. Tolerance
    basis: central differences with a 1e-6 relative step, measured agreement
    1e-9 to 1e-7 relative to the largest component.
    """
    disk = _ring(n_slices_los, gamma=jnp.array(gamma))
    weights = jnp.asarray(np.random.default_rng(0).random((16, 16)))
    base = jnp.array([getattr(disk, name).ravel()[0] for name in _FIT_FIELDS])
    start = jnp.concatenate([jnp.ones(len(_FIT_FIELDS)), jnp.array([1.0])])

    @jax.jit
    def loss(scale):
        values = [
            jnp.broadcast_to(scale[k] * base[k], getattr(disk, name).shape)
            for k, name in enumerate(_FIT_FIELDS)
        ]
        updated = eqx.tree_at(
            lambda d: [getattr(d, name) for name in _FIT_FIELDS], disk, values
        )
        image = radiance(
            updated,
            jnp.array(500.0),
            jnp.array(0.0),
            incl * scale[-1],
            90.0 - (jnp.array(_PA_DEG)),
            dist_pc=10.0,
            grid=_GRID,
        )
        return jnp.sum(weights * image)

    value = loss(start)
    grad = np.asarray(jax.jit(jax.grad(loss))(start)) / value
    step = 1e-6
    fd = np.array(
        [
            (loss(start.at[k].add(step)) - loss(start.at[k].add(-step))) / (2 * step)
            for k in range(start.size)
        ]
    ) / float(value)
    print(f"at {incl}: grad {grad}, max diff {np.abs(grad - fd).max():.2e}")
    assert np.all(np.isfinite(grad))
    assert np.all(np.abs(grad) > 1e-6), "a parameter lost its gradient"
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6 * np.abs(grad).max())


def test_fixed_vertical_profile_matches_the_general_density():
    """Fails if the power-free density differs.

    ``fixed_vertical_profile`` promises the same numbers to rounding for
    ``gamma = 2`` and ``beta = 1``: checked on the density over three decades
    in radius and on an image. A disk that sets the
    flag with another exponent must be rejected at construction.
    """
    rng = np.random.default_rng(0)
    r = jnp.asarray(10.0 ** rng.uniform(-1.0, 2.0, 200))
    z = jnp.asarray(rng.normal(0.0, 0.05, 200)) * r
    args = (r, z, 3.0, 4.5, -3.5, 0.2, 2.0, 1.0)
    np.testing.assert_allclose(
        two_power_law_density(*args, fixed_vertical_profile=True),
        two_power_law_density(*args),
        rtol=1e-12,
        atol=0.0,
    )
    general = _render(_ring(24), 45.0)
    fixed = _ring(24, fixed_vertical_profile=True)
    np.testing.assert_allclose(
        _render(fixed, 45.0), general, rtol=1e-10, atol=1e-13 * general.max()
    )
    with pytest.raises(ValueError, match="gamma = 2 and beta = 1"):
        _ring(24, gamma=jnp.array(1.5), fixed_vertical_profile=True)


def test_unusable_rule_is_rejected_at_construction():
    """Fails if a nonpositive node count surfaces only inside a trace."""
    for constructor in (_ring, _exovista):
        with pytest.raises(ValueError, match="must be positive"):
            constructor(0)


def _exovista(n_slices_los: int) -> ExovistaParametricDisk:
    """Ring at 20 AU with a halo to 80 AU on the 16-pixel grid, 3 AU pixels."""
    return ExovistaParametricDisk(
        r0_AU=jnp.array(20.0),
        dror=jnp.array(0.1),
        rinner_AU=jnp.array(2.0),
        hor=jnp.array(0.05),
        eta=jnp.array(1.0),
        g0=jnp.array(0.9),
        g1=jnp.array(0.6),
        g2=jnp.array(0.0),
        w0=jnp.array(0.7),
        w1=jnp.array(0.25),
        w2=jnp.array(0.05),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(80.0),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        Ag_grid=jnp.array([0.5, 0.5]),
        n_slices_los=n_slices_los,
        amplitude=DensityDust(
            level=jnp.array(3.0),
            profile_id="exovista-parametric",
            calibration_id="unit-profile",
        ),
    )


def test_exovista_rules_converge_to_the_reference():
    """Fails if the ExoVista scale height handed to the rules is wrong.

    The window needs ``h = sqrt(2) hor r`` (the Gaussian 1-sigma scale is
    ``hor r``). Reference: 513 nodes. Tolerance basis: measured errors
    2.1e-4 (96 nodes at 60 deg) and 2.8e-3 (96 at 90 deg), with margin.
    """
    reference = {
        i: _render(_exovista(513), i, sky_grid(16, 16, 0.3)) for i in (60.0, 90.0)
    }
    cases = (
        (96, 60.0, 1e-3),
        (96, 90.0, 2e-2),
    )
    for n_slices_los, incl, tolerance in cases:
        image = _render(_exovista(n_slices_los), incl, sky_grid(16, 16, 0.3))
        error = _peak_error(image, reference[incl])
        print(f"exovista {n_slices_los} at {incl}: {error:.2e}")
        assert error < tolerance

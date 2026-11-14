"""Line-of-sight quadrature rules of the parametric disks.

The rules share one integrand and differ in node placement, so each is
checked against a high-node reference, the reference against an independent
rule, and the gradients against central differences. Images are 16 x 16
pixels; each rule and node count compiles one small program.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.disk import ExovistaParametricDisk, GraterDisk
from skyscapes.disk._los import gauss_rule
from skyscapes.disk.grater import two_power_law_density

_PA_DEG = 30.0


def _ring(quadrature: str, n_slices_los: int, **overrides) -> GraterDisk:
    """Narrow ring at 0.7 AU (alpha = 5, h/r = 0.05) with tapered edges."""
    settings = dict(
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
        nx=16,
        ny=16,
        pixel_scale_arcsec=0.015,
        dist_pc=10.0,
        n_slices_los=n_slices_los,
        cutoff_width_AU=jnp.array(0.1),
        quadrature=quadrature,
    )
    settings.update(overrides)
    return GraterDisk(**settings)


@eqx.filter_jit
def _image(disk, incl_deg):
    return disk.surface_brightness(
        jnp.array(500.0), jnp.array(0.0), incl_deg, jnp.array(_PA_DEG)
    )


def _render(disk, incl_deg: float) -> np.ndarray:
    return np.asarray(_image(disk, jnp.array(incl_deg)))


def _peak_error(image: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(image - reference).max() / reference.max())


@pytest.fixture(scope="module")
def reference():
    """Gauss-Legendre images at 513 nodes, by inclination."""
    disk = _ring("legendre", 513)
    return {incl: _render(disk, incl) for incl in (45.0, 60.0, 88.0, 90.0)}


def test_gauss_rule_reproduces_hermite_and_laguerre():
    """Fails if the moments, recurrence or weight normalization are wrong.

    ``exp(-s^2)`` on the real line is Gauss-Hermite and ``exp(-s)`` on
    ``s > 0`` is Gauss-Laguerre. The NumPy path (a Python exponent) and the
    JAX path (an array exponent) must both reproduce them. Tolerance basis:
    measured 6e-15 (Hermite, 7 nodes) and 9e-14 relative (Laguerre, 5).
    """
    s_ref, w_ref = np.polynomial.hermite.hermgauss(7)
    for gamma in (2.0, jnp.array(2.0)):
        s, w = gauss_rule(gamma, 7)
        np.testing.assert_allclose(s, s_ref, atol=1e-12)
        np.testing.assert_allclose(w, w_ref, atol=1e-12)
    s_ref, w_ref = np.polynomial.laguerre.laggauss(5)
    for gamma in (1.0, jnp.array(1.0)):
        s, w = gauss_rule(gamma, 5, half_range=True)
        np.testing.assert_allclose(s, s_ref, rtol=1e-10)
        np.testing.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-14)


def test_reference_agrees_with_the_trapezoid_rule(reference):
    """Fails if the Gauss-Legendre window loses part of the integrand.

    The 513-node reference and the trapezoid rule at 1601 nodes share no
    nodes and no window, so agreement checks both. Tolerance basis: the
    trapezoid's own 1/n^2 error at 1601 nodes, measured 7e-6 to 1.2e-5.
    """
    trapezoid = _ring("trapezoid", 1601)
    for incl in (45.0, 90.0):
        error = _peak_error(_render(trapezoid, incl), reference[incl])
        print(f"reference vs trapezoid 1601 at {incl}: {error:.2e}")
        assert error < 1e-4


@pytest.mark.parametrize(
    ("quadrature", "n_slices_los", "incl", "tolerance"),
    [
        ("legendre", 48, 60.0, 1e-5),
        ("legendre", 96, 88.0, 1e-3),
        ("legendre", 96, 90.0, 1e-3),
        ("gauss", 7, 45.0, 2e-4),
        ("gauss", 7, 60.0, 2e-3),
        ("gauss_half", 4, 45.0, 5e-4),
    ],
)
def test_rule_converges_to_the_reference(
    reference, quadrature, n_slices_los, incl, tolerance
):
    """Fails if a rule's nodes or path weights are misplaced.

    Largest pixel error over the image peak, tapered edges. Tolerance basis:
    about ten times the errors measured on this grid (legendre 48 at 60 deg
    4e-8; legendre 96 at 88 deg 7e-5 and at 90 deg 4.5e-5; gauss 7 at 45 deg
    8e-6 and at 60 deg 1.8e-4; gauss_half 4 at 45 deg 6e-5). A 112-pixel
    image of the same ring samples the worst lines of sight and gives 2e-7,
    4.5e-4, 4.4e-4, 2e-5, 2.3e-4 and 6e-5.
    """
    disk = _ring(quadrature, n_slices_los)
    error = _peak_error(_render(disk, incl), reference[incl])
    print(f"{quadrature} {n_slices_los} at {incl}: {error:.2e}")
    assert error < tolerance


def test_gauss_validity_parameter():
    """Fails if ``gauss_rule_eps`` uses the wrong slope, thickness or angle unit.

    The ring has ``alpha = 5`` and ``h / r = 0.05``, so ``eps = 0.25`` at 45
    degrees. The ExoVista ring has edge slope ``1 / dror = 10`` and
    ``h / r = sqrt(2) hor``.
    """
    assert float(_ring("gauss", 7).gauss_rule_eps(jnp.array(45.0))) == pytest.approx(
        0.25, rel=1e-12
    )
    exo = _exovista("gauss", 7)
    assert float(exo.gauss_rule_eps(jnp.array(45.0))) == pytest.approx(
        10.0 * np.sqrt(2.0) * 0.05, rel=1e-12
    )


def test_half_range_rule_resolves_a_cusped_profile():
    """Fails if the half-range rule's reflection about the midplane is wrong.

    For ``gamma = 1.5`` the flared profile has a cusp at the midplane. Six
    half-range nodes per side must hold 1e-4 of the peak at 30 degrees and
    beat the symmetric rule with the same 12 nodes. Reference: trapezoid at
    3201 nodes. Tolerance basis: measured 2e-6 (half-range, 6 per side).
    """
    gamma = jnp.array(1.5)
    ref = _render(_ring("trapezoid", 3201, gamma=gamma), 30.0)
    half = _peak_error(_render(_ring("gauss_half", 6, gamma=gamma), 30.0), ref)
    symmetric = _peak_error(_render(_ring("gauss", 12, gamma=gamma), 30.0), ref)
    print(f"gamma 1.5: half-range {half:.2e}, symmetric {symmetric:.2e}")
    assert half < 1e-4
    assert half < symmetric


_FIT_FIELDS = ("sma_AU", "alpha_in", "alpha_out", "ksi0_AU", "gamma", "g_HG_grid")


@pytest.mark.parametrize(
    ("quadrature", "n_slices_los", "incl", "gamma"),
    [
        ("legendre", 32, 45.0, 2.0),
        ("legendre", 32, 88.0, 2.0),
        ("legendre", 32, 90.0, 2.0),
        ("gauss", 7, 45.0, 2.0),
        ("gauss_half", 5, 45.0, 1.5),
    ],
)
def test_gradients_match_central_differences(quadrature, n_slices_los, incl, gamma):
    """Fails if a rule's nodes or weights drop a dependence on a parameter.

    A weighted image sum is differentiated with respect to the shape
    parameters, the vertical exponent (which moves the Gauss nodes through
    the eigenproblem and sets the Gauss-Legendre window), the scattering
    asymmetry and the inclination, including exactly edge-on. Tolerance
    basis: central differences with a 1e-6 relative step, measured agreement
    1e-9 to 1e-7 relative to the largest component.
    """
    disk = _ring(quadrature, n_slices_los, gamma=jnp.array(gamma))
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
        image = updated.surface_brightness(
            jnp.array(500.0), jnp.array(0.0), incl * scale[-1], jnp.array(_PA_DEG)
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
    print(
        f"{quadrature} at {incl}: grad {grad}, max diff {np.abs(grad - fd).max():.2e}"
    )
    assert np.all(np.isfinite(grad))
    assert np.all(np.abs(grad) > 1e-6), "a parameter lost its gradient"
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6 * np.abs(grad).max())


def test_fixed_vertical_profile_matches_the_general_density():
    """Fails if the power-free density or the trace-time Gauss rule differs.

    ``fixed_vertical_profile`` promises the same numbers to rounding for
    ``gamma = 2`` and ``beta = 1``: checked on the density over three decades
    in radius and on images under both node families. A disk that sets the
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
    for quadrature, n_slices_los in (("legendre", 24), ("gauss", 7)):
        general = _render(_ring(quadrature, n_slices_los), 45.0)
        fixed = _ring(quadrature, n_slices_los, fixed_vertical_profile=True)
        np.testing.assert_allclose(
            _render(fixed, 45.0), general, rtol=1e-10, atol=1e-13 * general.max()
        )
    with pytest.raises(ValueError, match="gamma = 2 and beta = 1"):
        _ring("legendre", 24, gamma=jnp.array(1.5), fixed_vertical_profile=True)


@pytest.mark.parametrize(
    ("quadrature", "n_slices_los", "match"),
    [("simpson", 31, "must be one of"), ("gauss", 13, "at most 12")],
)
def test_unusable_rule_is_rejected_at_construction(quadrature, n_slices_los, match):
    """Fails if a bad rule name or node count surfaces only inside a trace."""
    with pytest.raises(ValueError, match=match):
        _ring(quadrature, n_slices_los)
    with pytest.raises(ValueError, match=match):
        _exovista(quadrature, n_slices_los)


def _exovista(quadrature: str, n_slices_los: int) -> ExovistaParametricDisk:
    """Ring at 20 AU with a halo to 80 AU on the 16-pixel grid, 3 AU pixels."""
    return ExovistaParametricDisk(
        r0_AU=jnp.array(20.0),
        dror=jnp.array(0.1),
        rinner_AU=jnp.array(2.0),
        hor=jnp.array(0.05),
        nzodis=jnp.array(3.0),
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
        nx=16,
        ny=16,
        pixel_scale_arcsec=0.3,
        dist_pc=10.0,
        n_slices_los=n_slices_los,
        quadrature=quadrature,
    )


def test_exovista_rules_converge_to_the_reference():
    """Fails if the ExoVista scale height handed to the rules is wrong.

    The rules need ``h = sqrt(2) hor r`` (the Gaussian 1-sigma scale is
    ``hor r``); a missing ``sqrt(2)`` narrows the window and moves the Gauss
    nodes. Reference: Gauss-Legendre at 513 nodes, the node counts of the
    ring tests. Tolerance basis: measured 2.1e-4 (legendre 96 at 60 deg),
    2.8e-3 (legendre 96 at 90 deg) and 3.7e-4 (gauss 7 at 30 deg), with a
    factor of 5 to 8 margin.
    """
    reference = {i: _render(_exovista("legendre", 513), i) for i in (30.0, 60.0, 90.0)}
    cases = (
        ("legendre", 96, 60.0, 1e-3),
        ("legendre", 96, 90.0, 2e-2),
        ("gauss", 7, 30.0, 3e-3),
    )
    for quadrature, n_slices_los, incl, tolerance in cases:
        image = _render(_exovista(quadrature, n_slices_los), incl)
        error = _peak_error(image, reference[incl])
        print(f"exovista {quadrature} {n_slices_los} at {incl}: {error:.2e}")
        assert error < tolerance

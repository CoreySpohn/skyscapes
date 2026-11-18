"""Physical disk images and demo elements survive the public frame change."""

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import deg2rad
from scipy.integrate import quad

from skyscapes.disk._los import los_integrate_scattered
from skyscapes.io import from_exovista

DEMO_ELEMENTS = [
    [
        0.387099,
        0.205625,
        1.130700508630296,
        2.195196872316272,
        4.447279311085948,
        3.0507260392391107,
    ],
    [
        0.722333199,
        0.00677323,
        1.061791299413483,
        2.1604296332674533,
        5.408989012265763,
        0.8761400741357597,
    ],
    [
        1.00000011,
        0.01671022,
        1.0471984070615956,
        2.0943949056238194,
        4.938259139713113,
        6.239952621485405,
    ],
    [
        1.52366231,
        0.09341233,
        1.0683109208837305,
        2.1224514048823475,
        2.7098146614955705,
        0.3386505464053476,
    ],
    [
        5.20336301,
        0.04839266,
        1.0431694355595715,
        2.120317184670284,
        3.386376544792743,
        0.3426787099262492,
    ],
    [
        9.53707032,
        0.0541506,
        1.03022203168904,
        2.140699654327287,
        4.730829565449819,
        5.542146144065518,
    ],
    [
        19.19126393,
        0.04716771,
        1.0508973908795851,
        2.1092947244136346,
        6.118510228872223,
        2.4825887001232516,
    ],
    [
        30.06896348,
        0.00858587,
        1.0268049688944991,
        2.1213289288422223,
        3.908464138773517,
        4.54058365372443,
    ],
]


@pytest.mark.parametrize(("inc", "pa"), [(35.0, 20.0), (60.0, 130.0), (125.0, -40.0)])
def test_disk_image_mirrors_stored_columns(inc, pa):
    """Fails if FITS columns or the near half reverse the physical disk image.

    Primitive reference: rotate each ray into the disk, integrate the Gaussian
    ring density times the HG phase divided by distance squared with adaptive
    quadrature on the outer cylinder chord. A quintic inner taper makes this
    a smooth orientation fixture; hard edges are covered by the GRaTeR tests.
    Allow 1e-4 of the peak for quadrature and exp(-16) vertical-tail truncation.
    """

    def density(r, z, valid):
        u = jnp.clip((r - 0.5) / 0.5, 0.0, 1.0)
        taper = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        return taper * jnp.exp(-((r - 2.0) ** 2 + (z / 0.2) ** 2))

    image = los_integrate_scattered(
        density,
        lambda cos_phi: (1.0 - 0.6**2) / (1.0 + 0.6**2 - 1.2 * cos_phi) ** 1.5,
        incl_deg=jnp.array(inc),
        position_angle_deg=jnp.array(90.0 - pa),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(4.0),
        scale_height_fn=lambda r: jnp.full_like(r, 0.2),
        east_AU=-jnp.broadcast_to(jnp.arange(5) - 2, (5, 5)),
        north_AU=jnp.broadcast_to((jnp.arange(5) - 2)[:, None], (5, 5)),
        n_slices_los=101,
    )
    ci, si = np.cos(inc * deg2rad), np.sin(inc * deg2rad)
    cp, sp = np.cos(pa * deg2rad), np.sin(pa * deg2rad)
    expected = np.zeros((5, 5))
    for row, column in np.ndindex(5, 5):
        east, north = 2.0 - column, row - 2.0
        u, v = east * cp + north * sp, -east * sp + north * cp
        chord = np.sqrt(4.0**2 - u**2)
        low, high = sorted(((-chord - v * ci) / si, (chord - v * ci) / si))
        points = [v * si / ci]
        if abs(u) < 0.5:
            inner = np.sqrt(0.5**2 - u**2)
            points.extend(((-inner - v * ci) / si, (inner - v * ci) / si))

        def integrand(depth, u=u, v=v):
            yd, zd = v * ci + depth * si, -v * si + depth * ci
            radius = np.hypot(u, yd)
            if radius < 0.5:
                return 0.0
            distance2 = u * u + v * v + depth * depth
            cosine = depth / np.sqrt(distance2)
            t = np.clip((radius - 0.5) / 0.5, 0.0, 1.0)
            taper = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
            density = taper * np.exp(-((radius - 2.0) ** 2 + (zd / 0.2) ** 2))
            phase = (1.0 - 0.6**2) / (1.0 + 0.6**2 - 1.2 * cosine) ** 1.5
            return density * phase / distance2

        expected[row, column] = quad(
            integrand,
            low,
            high,
            points=[x for x in points if low < x < high],
            epsabs=1e-14,
            epsrel=1e-12,
        )[0]
    np.testing.assert_allclose(image, expected, rtol=0, atol=1e-4 * expected.max())
    y, col = np.indices(image.shape) - 2
    x = -col  # Sky east is minus the FITS column offset.
    minor = -x * np.sin(np.radians(pa)) + y * np.cos(np.radians(pa))
    assert np.sum(image * minor) * np.cos(np.radians(inc)) > 0


def test_demo_elements_unchanged(fits_fixture):
    """Changing vector components preserves the demo's recovered orbital elements."""
    system = from_exovista(fits_fixture, load_disk=False)
    classical = system.dynamics.orbits.to_classical()
    actual = [
        [
            float(classical[key][index])
            for key in ("a_AU", "e", "i_rad", "W_rad", "w_rad", "M0_rad")
        ]
        for index in system.body_index
    ]
    np.testing.assert_allclose(actual, DEMO_ELEMENTS, rtol=1e-12, atol=1e-14)

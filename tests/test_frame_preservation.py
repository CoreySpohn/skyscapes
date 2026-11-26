"""Physical disk images and demo elements survive the public frame change."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.constants import deg2rad
from hwoutils.testing import ulps
from scipy.integrate import quad

from skyscapes.disk._los import los_integrate_scattered
from skyscapes.io import from_exovista


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


def test_demo_first_epoch_states_match_file(fits_fixture):
    """The demo system reproduces each file state relative to the star.

    Replaces the old header-element golden: the chain is built from the stored
    barycentric states, so each planet's first-epoch position and velocity
    relative to the star equal the file values after the axis change
    (west, north, toward) to (east, north, away) and AU/yr to AU/day.
    Each vector allows 32 roundings at its own norm for Jacobi differencing,
    frame construction, Kepler reconstruction and the rotation to sky axes.
    """
    system = from_exovista(fits_fixture, load_disk=False)
    with fits.open(fits_fixture, memmap=False) as hdul:
        star_ext = next(
            i
            for i, h in enumerate(hdul)
            if all(k in h.header for k in ("ANGDIAM", "MASS", "DIST"))
        )
        rows = [hdul[star_ext].data[0, 9:15].astype(float)] + [
            hdul[i].data[0, 9:15].astype(float)
            for i in range(star_ext + 1, len(hdul))
            if all(k in hdul[i].header for k in ("A", "E", "M", "R"))
        ]
    flip = np.array([-1.0, 1.0, -1.0])
    state = np.asarray(rows)
    position = state[:, :3] * flip
    velocity = state[:, 3:] * flip / 365.25
    t0 = system.dynamics.orbits.epoch_d[0]
    solved = system.dynamics.solve(jnp.asarray([t0]))
    got_r = np.asarray(solved.position_AU)[:, 0]
    got_v = np.asarray(solved.velocity_AU_per_d)[:, 0]
    # Chain rows follow the sorted order; body_index maps scene planets to them.
    chain = np.asarray(system.body_index)
    rel_r = got_r[1:][chain] - got_r[0]
    rel_v = got_v[1:][chain] - got_v[0]
    for recovered, stored in [(rel_r, position), (rel_v, velocity)]:
        for got, want in zip(recovered, stored[1:] - stored[0], strict=True):
            np.testing.assert_allclose(
                got, want, rtol=0, atol=float(ulps(32, np.linalg.norm(want)))
            )

"""Shared pytest fixtures for skyscapes tests."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
from hwoutils.constants import AU2m, G, d2s, pc2AU, rad2arcsec
from hypothesis import settings
from jax import config

# ExoJAX refuses to initialize in JAX's default 32-bit mode (raises on
# import of MdbHitran / Art* engines). Flip x64 before any test imports
# touch JAX state so the slow integration test can run on CI.
config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from optixstuff.grids import PixelGrid, SkyOrientation  # noqa: E402

from skyscapes.datasets import fetch_scene  # noqa: E402
from skyscapes.scene import FlatStar  # noqa: E402

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


def two_body_reference(
    time_d,
    *,
    a_AU,
    e,
    W_rad,
    i_rad,
    w_rad,
    M0_rad,
    epoch_d=0.0,
    star_mass_kg,
    planet_mass_kg=0.0,
    dist_pc=1.0,
):
    """Independent float64 elliptic two-body states and sky observables.

    Solve E - e sin(E) = M by Newton iteration to 1e-14 radians, then
    rotate perifocal positions and velocities by (W, i, w). Omega is
    measured north through east; the third coordinate points away.
    Phase is atan2(projected radius, away), PA is east of north, and
    stellar RV is receding-positive. Masses are kg, time offsets are days,
    and vector arrays have shape (K, 3, T); scalar observables are (K, T).
    No orbit engine or scene observable supplies the reference calculation.
    """
    a, ecc, node, inc, peri, anomaly, epoch, ms, mp = np.broadcast_arrays(
        *(
            np.atleast_1d(np.asarray(value, dtype=np.float64))
            for value in (
                a_AU,
                e,
                W_rad,
                i_rad,
                w_rad,
                M0_rad,
                epoch_d,
                star_mass_kg,
                planet_mass_kg,
            )
        )
    )
    times = np.atleast_1d(np.asarray(time_d, dtype=np.float64))
    motion = np.sqrt(G * (ms + mp) / a**3)
    mean = (anomaly[:, None] + motion[:, None] * (times - epoch[:, None])) % (2 * np.pi)
    eccentric = np.where(ecc[:, None] < 0.8, mean, np.pi)
    for _ in range(100):
        step = (eccentric - ecc[:, None] * np.sin(eccentric) - mean) / (
            1 - ecc[:, None] * np.cos(eccentric)
        )
        eccentric -= step
        if np.all(np.abs(step) <= 1e-14):
            break
    else:
        raise AssertionError("Two-body reference Newton iteration did not converge")

    cosE, sinE = np.cos(eccentric), np.sin(eccentric)
    root = np.sqrt(1 - ecc**2)
    x = a[:, None] * (cosE - ecc[:, None])
    y = (a * root)[:, None] * sinE
    rate = motion[:, None] / (1 - ecc[:, None] * cosE)
    vx = -a[:, None] * sinE * rate
    vy = (a * root)[:, None] * cosE * rate

    # Rotate first in the orbital plane, incline about the node, then put
    # the ascending node at (sin W, cos W, 0) in the east/north sky plane.
    def rotate(x, y):
        along = np.cos(peri)[:, None] * x - np.sin(peri)[:, None] * y
        across = np.sin(peri)[:, None] * x + np.cos(peri)[:, None] * y
        return np.stack(
            [
                np.sin(node)[:, None] * along
                + (np.cos(node) * np.cos(inc))[:, None] * across,
                np.cos(node)[:, None] * along
                - (np.sin(node) * np.cos(inc))[:, None] * across,
                np.sin(inc)[:, None] * across,
            ],
            axis=1,
        )

    position, velocity = rotate(x, y), rotate(vx, vy)
    projected = np.hypot(position[:, 0], position[:, 1])
    distance = np.linalg.norm(position, axis=1)
    scale = rad2arcsec / (np.atleast_1d(np.asarray(dist_pc, np.float64)) * pc2AU)
    offsets = position[:, :2].transpose(1, 0, 2) * scale[None, :, None]
    fraction = -mp / (ms + mp)
    angle = np.rad2deg(np.arctan2(position[:, 0], position[:, 1])) % 360.0
    return SimpleNamespace(
        position_AU=position,
        velocity_AU_per_d=velocity,
        phase_angle_rad=np.arctan2(projected, position[:, 2]),
        distance_AU=distance,
        position_arcsec=offsets,
        separation_arcsec=np.hypot(offsets[0], offsets[1]),
        position_angle_deg=np.where(angle == 360.0, 0.0, angle),
        star_reflex_arcsec=offsets * fraction[None, :, None],
        star_radial_velocity_m_per_s=fraction[:, None] * velocity[:, 2] * (AU2m / d2s),
    )


@pytest.fixture(scope="session")
def fits_fixture() -> str:
    """Path to the canonical ExoVista demo FITS (cached via pooch)."""
    return fetch_scene()


# Shared sky sampling for disk shape and quadrature tests.
def sky_grid(nx=51, ny=51, step=0.2):
    """North-up sampling; geometric centers also cover even images."""
    return PixelGrid.regular((ny, nx), step, "arcsec", orientation=SkyOrientation(0.0))


def radiance(
    disk,
    wavelength_nm,
    time_d=0.0,
    inclination_deg=60.0,
    position_angle_deg=90.0,
    *,
    grid,
    dist_pc=10.0,
):
    """Unit illumination at 1 AU isolates profile/LOS tests from photometry.

    Phi_observer = (d_pc*pc2AU)^-2 makes Phi_incident(1 AU)=1.
    Absolute photometry has separate primitive anchors in test_disk.py.
    """
    star = FlatStar(jnp.asarray((dist_pc * pc2AU) ** -2))
    return disk.photon_radiance(
        jnp.asarray(wavelength_nm),
        time_d,
        star=star,
        dist_pc=dist_pc,
        inclination_deg=inclination_deg,
        position_angle_deg=position_angle_deg,
        grid=grid,
    ).data

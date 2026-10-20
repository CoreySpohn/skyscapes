"""Imported midplane angles agree with the independently rendered FITS image."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyscapes import viz
from skyscapes.disk import GraterDisk
from skyscapes.io import from_exovista
from skyscapes.io.exovista import _midplane_to_east_north
from skyscapes.viz._geometry import disk_axes_sky


def _major_axis_deg(x, y, weights):
    """Central second-moment major axis from east toward north, modulo 180."""
    x = x - np.average(x, weights=weights)
    y = y - np.average(y, weights=weights)
    xx = np.average(x * x, weights=weights)
    yy = np.average(y * y, weights=weights)
    xy = np.average(x * y, weights=weights)
    return np.degrees(0.5 * np.arctan2(2 * xy, xx - yy)) % 180


@pytest.fixture(scope="module")
def midplane_measurements(fits_fixture):
    """Measure a forward-scattering ring on the demo cube's exact pixel grid."""
    system = from_exovista(fits_fixture)
    loaded = system.disk
    wavelength_nm = loaded.wavelengths_nm[0]
    args = tuple(
        jnp.asarray(value)
        for value in (
            wavelength_nm,
            2451545.0,
            system.midplane_inc_deg,
            system.midplane_pa_deg,
        )
    )
    reference = np.asarray(loaded.surface_brightness(*args), dtype=float)
    ny, nx = reference.shape
    # The suite's two-power-law Grater ring, sized to fit this field so
    # clipping does not set its moments. Only orientation is compared;
    # this is not a fit to ExoVista's density or scattering function.
    radius_AU = 0.25 * min(nx, ny) * loaded.pixel_scale_arcsec * system.star.dist_pc
    parametric = GraterDisk(
        sma_AU=jnp.asarray(radius_AU),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-5.0),
        ksi0_AU=jnp.asarray(0.02 * radius_AU),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.asarray(0.1 * radius_AU),
        rmax_AU=jnp.asarray(1.8 * radius_AU),
        wavelengths_nm=jnp.array([wavelength_nm, loaded.wavelengths_nm[-1]]),
        g_HG_grid=jnp.array([0.3, 0.3]),
        Ag_grid=jnp.array([0.5, 0.5]),
        nx=nx,
        ny=ny,
        pixel_scale_arcsec=loaded.pixel_scale_arcsec,
        dist_pc=system.star.dist_pc,
        n_slices_los=31,
    )
    rendered = np.asarray(parametric.surface_brightness(*args), dtype=float)
    for image in (reference, rendered):
        assert np.all(np.isfinite(image)) and np.all(image >= 0)
        assert image.sum() > 0
    row, col = np.indices(reference.shape, dtype=float)
    east, north = col - (nx - 1) / 2, row - (ny - 1) / 2
    reference_angle = _major_axis_deg(east, north, reference)
    rendered_angle = _major_axis_deg(east, north, rendered)
    # Fix the signed minor direction from the reference image alone, so
    # changing the imported PA cannot also change the comparison axis.
    angle_rad = np.radians(reference_angle)
    minor_offset = -np.sin(angle_rad) * east + np.cos(angle_rad) * north
    reference_first = np.average(minor_offset, weights=reference)
    rendered_first = np.average(minor_offset, weights=rendered)

    fig, ax = plt.subplots()
    try:
        viz.plot_system(
            system,
            2451545.0,
            ax=ax,
            disk_radii_AU=(float(parametric.rmin_AU), float(parametric.rmax_AU)),
        )
        outline = next(line for line in ax.lines if line.get_gid() == "disk/outer")
        # Drop the duplicate closing point before taking second moments.
        x, y = np.asarray(outline.get_data())[:, :-1]
        outline_angle = _major_axis_deg(x, y, np.ones_like(x))
    finally:
        plt.close(fig)
    return (
        reference_angle,
        rendered_angle,
        reference_first,
        rendered_first,
        outline_angle,
    )


@pytest.mark.parametrize("measurement", ["render_axis", "near_side", "outline_axis"])
def test_exovista_midplane_matches_image(midplane_measurements, measurement):
    """The imported plane AND forward-scattering side match the demo image.

    The brightness-weighted demo axis is 147.34 deg, while the ideal
    parametric ring is 150.00 deg and the outline is 150 deg. A 3 deg
    tolerance covers that 2.66 deg offset from disk structure; the old
    30 deg orientation misses by 62.66 deg. The signed minor-axis first
    moments are -10.625 px (FITS) and -24.833 px (ring). These are measured
    about the star, not about the centroid; only the signs must agree.
    """
    reference, rendered, reference_first, rendered_first, outline = (
        midplane_measurements
    )
    if measurement == "near_side":
        # A floor keeps a near-symmetric residual from passing on sign alone.
        assert min(abs(reference_first), abs(rendered_first)) > 1.0
        assert reference_first * rendered_first > 0, (
            f"minor-axis first moments: FITS={reference_first:.3f} px, "
            f"parametric={rendered_first:.3f} px"
        )
    else:
        actual = rendered if measurement == "render_axis" else outline
        difference = abs((actual - reference + 90) % 180 - 90)
        assert difference < 3.0, (
            f"{measurement}: FITS={reference:.3f} deg, actual={actual:.3f} deg, "
            f"difference={difference:.3f} deg"
        )


@pytest.mark.parametrize("inc_deg", [30.0, 60.0, 120.0, 150.0])
@pytest.mark.parametrize("pa_deg", [-30.0, 0.0, 30.0, 213.0, 390.0])
def test_midplane_conversion_reflects_plane_and_near_side(inc_deg, pa_deg):
    """Reflect ExoVista's basis across x, including inclinations above 90.

    The nodes may reverse along their line, but the near vector must match
    in all three coordinates. The supplementary-inclination representation
    must describe the same plane AND near half. Tolerance: float64 trig.
    """
    inc, pa = np.radians([inc_deg, pa_deg])
    reflected_nodes = np.array([-np.cos(pa), np.sin(pa), 0.0])
    reflected_near = np.array(
        [np.cos(inc) * np.sin(pa), np.cos(inc) * np.cos(pa), np.sin(inc)]
    )
    converted_inc, converted_pa = _midplane_to_east_north(inc_deg, pa_deg)
    assert converted_inc == inc_deg
    assert 0.0 <= converted_pa < 360.0
    nodes, near, normal = disk_axes_sky(converted_inc, converted_pa)
    np.testing.assert_allclose(-nodes, reflected_nodes, atol=1e-14)
    np.testing.assert_allclose(near, reflected_near, atol=1e-14)
    equivalent_nodes, equivalent_near, equivalent_normal = disk_axes_sky(
        180.0 - converted_inc, converted_pa + 180.0
    )
    np.testing.assert_allclose(equivalent_nodes, -nodes, atol=1e-14)
    np.testing.assert_allclose(equivalent_near, near, atol=1e-14)
    np.testing.assert_allclose(equivalent_normal, -normal, atol=1e-14)

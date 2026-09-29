"""Contract tests for the skyscapes.viz views.

The geometry checks are independent of the plotting code path: projected
ring shapes are compared with the analytic ``|cos i|`` and with the
second moments of the disk that the line-of-sight kernel renders, the
near side with the kernel's forward-scattering asymmetry, and the observer
side with orbix's phase angle (measured from the ``+z`` observer axis).
"""

from __future__ import annotations

import re

import eyepiece
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

import skyscapes.viz as viz
from skyscapes import Scene, System
from skyscapes.disk import ExovistaDisk, GraterDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet

matplotlib.use("Agg")

SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
DIST_PC = 10.0


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates."""
    yield
    plt.close("all")


def make_disk(g=0.4, nx=81, rmin=1.5, rmax=5.0):
    """A GraterDisk spanning ``[rmin, rmax]`` AU at 10 pc."""
    return GraterDisk(
        sma_AU=jnp.array(3.0),
        alpha_in=jnp.array(5.0),
        alpha_out=jnp.array(-3.0),
        ksi0_AU=jnp.array(0.1),
        gamma=jnp.array(2.0),
        beta=jnp.array(1.0),
        rmin_AU=jnp.array(rmin),
        rmax_AU=jnp.array(rmax),
        wavelengths_nm=jnp.array([400.0, 1000.0]),
        g_HG_grid=jnp.array([g, g]),
        Ag_grid=jnp.array([0.3, 0.3]),
        nx=nx,
        ny=nx,
        pixel_scale_arcsec=1.3 * rmax / DIST_PC / (nx / 2),
        dist_pc=DIST_PC,
        n_slices_los=21,
    )


def make_planets():
    """Two inclined, eccentric planets (not coplanar with any disk)."""
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0, 2.5]),
        e=jnp.array([0.05, 0.2]),
        W_rad=jnp.array([0.4, 0.6]),
        i_rad=jnp.deg2rad(jnp.array([55.0, 62.0])),
        w_rad=jnp.array([0.3, 1.2]),
        M0_rad=jnp.array([0.0, 2.0]),
        t0_d=jnp.array([0.0, 0.0]),
    )
    return Planet(
        Rp_Rearth=jnp.array([1.0, 3.0]),
        Mp_Mearth=jnp.array([1.0, 10.0]),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3, 0.3])),
    )


def make_system(incl=60.0, pa=30.0, disk=None, planets=True):
    """A star, two planets and a disk at the given midplane orientation."""
    star = FlatStar(Ms_kg=Msun2kg, dist_pc=DIST_PC, flux_phot_per_nm_m2=1e9)
    return System(
        star=star,
        planets=(make_planets(),) if planets else (),
        trig_solver=SOLVER,
        disk=make_disk() if disk is None else disk,
        midplane_inc_deg=incl,
        midplane_pa_deg=pa,
    )


def flat(artists):
    """Every artist in a result's artists dict, flattened."""
    out = []
    for value in artists.values():
        out.extend(value if isinstance(value, list) else [value])
    return out


def by_gid(result):
    """The result's artists keyed by gid."""
    return {
        a.get_gid(): a
        for a in flat(result.artists)
        if hasattr(a, "get_gid") and a.get_gid()
    }


def state(artist):
    """A comparable snapshot of what an artist currently draws."""
    from matplotlib.collections import PathCollection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from matplotlib.text import Annotation, Text

    if isinstance(artist, Line2D):
        return np.asarray(artist.get_xydata(), float).copy()
    if isinstance(artist, PathCollection):
        return np.asarray(artist.get_offsets(), float).copy()
    if isinstance(artist, Annotation):
        return (tuple(artist.xy), tuple(artist.xyann), artist.get_text())
    if isinstance(artist, Text):
        return (tuple(artist.get_position()), artist.get_text())
    if isinstance(artist, Polygon):
        return np.asarray(artist.get_xy(), float).copy()
    if isinstance(artist, AxesImage):
        return np.asarray(artist.get_array(), float).copy()
    raise TypeError(type(artist))


def same(a, b):
    """Whether two snapshots are identical."""
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a == b


def principal_axes(points):
    """Axis ratio and major-axis angle [deg, mod 180] of a centered cloud."""
    p = np.asarray(points, float)
    p = p - p.mean(axis=0)
    evals, evecs = np.linalg.eigh(p.T @ p / len(p))
    major = evecs[:, 1]
    angle = np.degrees(np.arctan2(major[1], major[0])) % 180.0
    return np.sqrt(evals[0] / evals[1]), angle


def image_axes(image):
    """Flux-weighted axis ratio proxy and major-axis angle of an image.

    Column index is the sky ``x`` (RA offset) axis and row index ``y``.
    """
    image = np.asarray(image, float)
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2)
    w = image / image.sum()
    cov = np.array(
        [
            [(w * x * x).sum(), (w * x * y).sum()],
            [(w * x * y).sum(), (w * y * y).sum()],
        ]
    )
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    return np.sqrt(evals[0] / evals[1]), np.degrees(
        np.arctan2(major[1], major[0])
    ) % 180


def angle_between(u, v):
    """Angle between two 2D vectors [deg]."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def arrow_vector(annotation):
    """Tail-to-head vector of an arrow annotation."""
    return np.asarray(annotation.xy, float) - np.asarray(annotation.xyann, float)


def angle_diff_mod180(a, b):
    """Smallest difference between two axis angles [deg]."""
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


# --------------------------------------------------------------------------
# The return contract and global state
# --------------------------------------------------------------------------

VIEWS = {
    "system_sky": lambda s, ax: viz.plot_system(
        s, 0.0, ax=ax, track_t_jd=jnp.linspace(0, 900, 50)
    ),
    "system_side": lambda s, ax: viz.plot_system(s, 0.0, view="side", ax=ax),
    "disk_image": lambda s, ax: viz.plot_disk_image(
        s, wavelength_nm=550.0, ax=ax, grain_radius_AU=3.0
    ),
    "disk_geometry": lambda s, ax: viz.plot_disk_geometry(s, ax=ax, thickness_AU=0.5),
    "zodi_top": lambda s, ax: viz.plot_local_zodi_geometry(30.0, 135.0, ax=ax),
    "zodi_side": lambda s, ax: viz.plot_local_zodi_geometry(
        30.0, 135.0, view="side", ax=ax
    ),
    "scattering_angle": lambda s, ax: viz.plot_scattering_angle(
        [1.0, 0.0], [0.5, 0.8], ax=ax
    ),
}


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_view_returns_plot_result_on_the_supplied_axes(name):
    """Each view draws on the handed axes (or insets of it) and nothing else."""
    system = make_system()
    _, ax = plt.subplots()
    figs_before = plt.get_fignums()
    rc_before = dict(matplotlib.rcParams)

    result = VIEWS[name](system, ax)

    assert isinstance(result, eyepiece.PlotResult)
    assert result.ax is ax
    assert set(result.artists) <= eyepiece.ARTIST_KEYS
    for artist in flat(result.artists):
        # A colorbar is not an artist: its own axes is the thing to place.
        owner = artist.ax if hasattr(artist, "mappable") else artist.axes
        assert owner is ax or owner in ax.child_axes, (name, artist)
    assert plt.get_fignums() == figs_before
    rc_after = dict(matplotlib.rcParams)
    assert [k for k in rc_before if rc_before[k] != rc_after[k]] == []


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_view_creates_a_figure_when_no_axes_is_given(name):
    """``ax=None`` gives a fresh figure holding the view."""
    result = VIEWS[name](make_system(), None)
    assert isinstance(result, eyepiece.PlotResult)
    assert result.fig is result.ax.figure


def test_views_accept_a_scene():
    """A Scene is unpacked to its System by every system-level view."""
    scene = Scene(system=make_system())
    viz.plot_system(scene, 0.0)
    viz.plot_disk_image(scene, wavelength_nm=550.0)
    viz.plot_disk_geometry(scene)


def test_disk_image_keeps_its_slot_in_a_grid():
    """Drawing into one of two slots leaves the slots equal in width."""
    fig, axes = plt.subplots(1, 2, layout="constrained")
    viz.plot_disk_image(make_system(), wavelength_nm=550.0, ax=axes[0])
    fig.canvas.draw()
    w0 = axes[0].get_position(original=True).width
    w1 = axes[1].get_position(original=True).width
    assert w0 == pytest.approx(w1)


def test_colors_follow_the_active_mode_at_call_time():
    """The star role is resolved per call, so a mode switch is honored."""
    import hwostyle

    with hwostyle.light():
        light = by_gid(viz.plot_system(make_system(), 0.0))["star"].get_facecolor()
    with hwostyle.dark():
        dark = by_gid(viz.plot_system(make_system(), 0.0))["star"].get_facecolor()
    assert not np.allclose(light, dark)


def test_source_styles_color_the_planets():
    """A declared cast colors the planets it names; others take the role."""
    styles = eyepiece.SourceStyles(["star", "b", "c"])
    gids = by_gid(viz.plot_system(make_system(), 0.0, styles=styles))
    from matplotlib.colors import to_rgba

    assert np.allclose(
        gids["planet/c"].get_facecolor()[0], to_rgba(styles["c"]["color"])
    )


# --------------------------------------------------------------------------
# Geometry pinned against independent references
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("incl", "pa"), [(60.0, 30.0), (35.0, 120.0), (120.0, -40.0)])
def test_projected_ring_axis_ratio_is_abs_cos_i(incl, pa):
    """The drawn outer ring has axis ratio |cos i| and its major axis at pa.

    The position angle convention checked is the library's: the line of
    nodes lies at ``pa`` from the ``+x`` (RA offset) axis toward ``+y``.
    """
    result = viz.plot_system(make_system(incl, pa), 0.0)
    ring = by_gid(result)["disk/outer"].get_xydata()[:-1]
    ratio, angle = principal_axes(ring)
    assert ratio == pytest.approx(abs(np.cos(np.radians(incl))), abs=1e-9)
    assert angle_diff_mod180(angle, pa) < 1e-6


@pytest.mark.parametrize(("incl", "pa"), [(60.0, 30.0), (50.0, 120.0)])
def test_ring_orientation_matches_the_rendered_disk(incl, pa):
    """The outline's major axis matches the kernel's rendered image moments."""
    system = make_system(incl, pa)
    image = system.disk.surface_brightness(
        jnp.array(550.0), jnp.array(0.0), jnp.array(incl), jnp.array(pa)
    )
    _, image_angle = image_axes(image)
    result = viz.plot_system(system, 0.0)
    _, ring_angle = principal_axes(by_gid(result)["disk/outer"].get_xydata()[:-1])
    assert angle_diff_mod180(ring_angle, image_angle) < 0.5


def test_disk_image_pixels_land_at_their_sky_coordinates():
    """Column index maps to +RA offset, drawn increasing to the left."""
    image = np.full((21, 21), 1e-8)
    image[15, 17] = 1.0  # row 15 is +y, column 17 is +x
    result = viz.plot_disk_image(image, pixel_scale_arcsec=0.1)
    left, right, bottom, top = result.artists["image"].get_extent()
    assert left < right and bottom < top
    assert result.ax.xaxis_inverted()
    # The bright pixel's center, from the extent, is at +x, +y.
    x = left + (17 + 0.5) * (right - left) / 21
    y = bottom + (15 + 0.5) * (top - bottom) / 21
    assert x > 0.0 and y > 0.0
    center = result.ax.transData.transform((0.0, 0.0))
    spot = result.ax.transData.transform((x, y))
    assert spot[0] < center[0], "positive RA offset must be drawn to the left"
    assert spot[1] > center[1]


def test_disk_image_outline_matches_the_rendered_image():
    """The outline drawn on the image follows the image's own orientation."""
    result = viz.plot_disk_image(make_system(55.0, 70.0), wavelength_nm=550.0)
    _, image_angle = image_axes(result.artists["image"].get_array())
    _, ring_angle = principal_axes(by_gid(result)["disk/outer"].get_xydata()[:-1])
    assert angle_diff_mod180(ring_angle, image_angle) < 0.5


@pytest.mark.parametrize("pa", [30.0, 150.0])
def test_near_side_is_where_the_kernel_scatters_forward(pa):
    """The labeled near side is the bright half of a forward-scattering disk."""
    system = make_system(60.0, pa, disk=make_disk(g=0.6))
    image = np.asarray(
        system.disk.surface_brightness(
            jnp.array(550.0), jnp.array(0.0), jnp.array(60.0), jnp.array(pa)
        )
    )
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2)
    centroid = np.array([(image * x).sum(), (image * y).sum()]) / image.sum()

    sky = by_gid(viz.plot_system(system, 0.0))
    label = np.asarray(sky["label/near_side"].get_position())
    assert label @ centroid > 0.0

    side = by_gid(viz.plot_system(system, 0.0, view="side"))
    z_near = side["disk/near"].get_xdata()
    assert np.all(z_near > 0.0), "the near half must lie toward the observer"


def test_side_view_observer_is_orbix_plus_z():
    """The observer arrow points to +z, and a planet in front is drawn there.

    orbix measures its phase angle from the +z observer axis, so a planet
    with phase angle below 90 degrees lies between the star and observer.
    """
    system = make_system()
    t = jnp.array([0.0, 200.0, 400.0])
    result = viz.plot_system(system, 0.0, view="side")
    arrow = arrow_vector(by_gid(result)["observer"])
    assert arrow[0] > 0.0 and arrow[1] == pytest.approx(0.0)

    for t_now in np.asarray(t):
        _, phase, _ = system.planets[0].propagate(
            SOLVER, jnp.array([t_now]), star=system.star
        )
        in_front = np.cos(np.asarray(phase)[:, 0]) > 0.0
        side = by_gid(viz.plot_system(system, float(t_now), view="side"))
        for k, name in enumerate(["b", "c"]):
            z = side[f"planet/{name}"].get_offsets()[0, 0]
            assert (z > 0.0) == bool(in_front[k])


@pytest.mark.parametrize("incl", [20.0, 60.0, 75.0])
def test_scattering_angle_on_the_minor_axis_is_90_minus_i(incl):
    """A near-side grain on the minor axis scatters at 90 - i; far side 90 + i."""
    system = make_system(incl, 30.0)
    for sign, expected in ((1.0, 90.0 - incl), (-1.0, 90.0 + incl)):
        result = viz.plot_disk_geometry(system, grain_radius_AU=sign * 3.0)
        gids = by_gid(result)
        drawn = angle_between(
            arrow_vector(gids["incident"]), arrow_vector(gids["scattered"])
        )
        assert drawn == pytest.approx(expected, abs=1e-6)
        readout = gids["inset/scattering_angle/value"].get_text()
        assert f"{expected:.0f}" in readout


def test_geometry_panel_places_the_midplane_at_the_inclination():
    """The midplane makes angle i with the sky plane, near half at +z."""
    for incl in (10.0, 45.0, 80.0):
        gids = by_gid(viz.plot_disk_geometry(make_system(incl, 0.0)))
        z, u = gids["disk/near"].get_xydata().T
        tilt = np.degrees(np.arctan2(z[1] - z[0], u[1] - u[0]))
        assert tilt == pytest.approx(incl, abs=1e-9)
        assert np.all(z > 0.0)


def test_zodi_sightline_angles_match_the_inputs():
    """Top view: projected look at Delta lambda from the Sun; side: at beta."""
    beta, dlon = 25.0, 110.0
    top = by_gid(viz.plot_local_zodi_geometry(beta, dlon))
    (x0, y0), (x1, y1) = top["sightline"].get_xydata()
    look = np.array([x1 - x0, y1 - y0])
    assert angle_between(look, [-1.0, 0.0]) == pytest.approx(dlon, abs=1e-9)

    side = by_gid(viz.plot_local_zodi_geometry(beta, dlon, view="side"))
    (x0, y0), (x1, y1) = side["sightline"].get_xydata()
    assert np.degrees(np.arctan2(y1 - y0, x1 - x0)) == pytest.approx(beta, abs=1e-9)


def test_zodi_readout_separates_elongation_from_longitude_difference():
    """Off the ecliptic the 3D elongation is not Delta lambda.

    Looking 60 degrees above the ecliptic, directly away from the Sun in
    longitude, the Sun is 120 degrees away on the sky, not 180.
    """
    text = by_gid(viz.plot_local_zodi_geometry(60.0, 180.0))["label/angles"]
    assert "solar elongation = 120" in text.get_text()


def test_zodi_grain_scattering_angle_known_answer():
    """In-ecliptic look at 90 deg, grain 1 AU out: the grain scatters at 135."""
    gids = by_gid(viz.plot_local_zodi_geometry(0.0, 90.0, grain_distance_AU=1.0))
    assert "135" in gids["inset/scattering_angle/value"].get_text()
    assert "45" in gids["inset/illumination_angle/value"].get_text()


# --------------------------------------------------------------------------
# Updaters change only what they declare
# --------------------------------------------------------------------------


def _all_states(result):
    """Snapshot of every gid-carrying artist on the axes and its insets."""
    return {gid: state(a) for gid, a in by_gid(result).items()}


def _check_update(result, update_args, restore_args, fixed, moved):
    """Run an update, check fixed/moved artists, then restore exactly."""
    ax = result.ax
    n_children = len(ax.get_children())
    n_inset_children = [len(c.get_children()) for c in ax.child_axes]
    before = _all_states(result)

    result.update(*update_args)
    after = _all_states(result)
    for gid in fixed:
        assert same(before[gid], after[gid]), f"{gid} moved"
    for gid in moved:
        assert not same(before[gid], after[gid]), f"{gid} did not move"
    assert len(ax.get_children()) == n_children
    assert [len(c.get_children()) for c in ax.child_axes] == n_inset_children

    result.update(*restore_args)
    restored = _all_states(result)
    for gid, snap in before.items():
        assert same(snap, restored[gid]), f"{gid} not restored"


def test_disk_geometry_update_moves_only_inclination_artists():
    """``update(incl)`` moves the midplane, rays and arcs, not the frame."""
    result = viz.plot_disk_geometry(make_system(60.0), thickness_AU=0.5)
    _check_update(
        result,
        (25.0,),
        (60.0,),
        fixed=["star", "sky_plane", "observer", "label/observer", "label/sky_plane"],
        moved=[
            "disk/near",
            "disk/far",
            "disk/layer_near",
            "grain",
            "incident",
            "sightline",
            "sightline/path",
            "inclination/label",
            "inset/scattering_angle/value",
        ],
    )


def test_disk_image_update_moves_image_and_outline_only():
    """``update(image, incl_deg=...)`` swaps the data and moves the outline."""
    system = make_system(60.0, 30.0)
    result = viz.plot_disk_image(system, wavelength_nm=550.0, grain_radius_AU=3.0)
    frame = system.disk.surface_brightness(
        jnp.array(550.0), jnp.array(0.0), jnp.array(30.0), jnp.array(30.0)
    )
    original = result.artists["image"].get_array().copy()
    before = _all_states(result)
    result.update(frame, incl_deg=30.0)
    after = _all_states(result)
    assert same(before["star"], after["star"])
    assert not same(before["disk/outer"], after["disk/outer"])
    assert not same(before["sightline"], after["sightline"])
    assert not np.array_equal(original, result.artists["image"].get_array())


def test_zodi_update_moves_only_look_artists():
    """``update(beta, dlon)`` moves the sightline, not the observer or orbit."""
    result = viz.plot_local_zodi_geometry(30.0, 135.0)
    _check_update(
        result,
        (10.0, 60.0),
        (30.0, 135.0),
        fixed=["sun", "observer", "observer_orbit", "label/observer", "label/sun"],
        moved=["sightline", "grain", "incident", "label/angles", "look_angle"],
    )


# --------------------------------------------------------------------------
# Inputs the views refuse or degrade on
# --------------------------------------------------------------------------


def test_pre_rendered_disk_draws_no_invented_outline():
    """An ExovistaDisk declares no radii: no rings unless the caller gives them."""
    wl = jnp.linspace(400.0, 1000.0, 3)
    cube = jnp.full((3, 8, 8), 1e-7)
    disk = ExovistaDisk(pixel_scale_arcsec=0.1, wavelengths_nm=wl, contrast_cube=cube)
    system = make_system(disk=disk)
    assert "disk/outer" not in by_gid(viz.plot_system(system, 0.0))
    given = by_gid(viz.plot_system(system, 0.0, disk_radii_AU=(1.0, 4.0)))
    assert "disk/outer" in given
    with pytest.raises(ValueError, match="radii_AU"):
        viz.plot_disk_geometry(system)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda s: viz.plot_system(s, 0.0, view="top"), "view"),
        (lambda s: viz.plot_system(s), "t_jd"),
        (lambda s: viz.plot_disk_image(np.ones((4, 4))), "pixel_scale"),
        (lambda s: viz.plot_disk_image(s.disk), "wavelength_nm"),
        (lambda s: viz.plot_disk_image(np.ones(4), pixel_scale_arcsec=0.1), "2D"),
        (lambda s: viz.plot_disk_geometry(s, incl_deg=200.0), "incl_deg"),
        (lambda s: viz.plot_local_zodi_geometry(95.0, 10.0), "ecliptic_lat"),
        (lambda s: viz.plot_local_zodi_geometry(10.0, 190.0), "solar_lon"),
        (lambda s: viz.plot_local_zodi_geometry(10.0, 10.0, view="x"), "view"),
    ],
)
def test_bad_inputs_raise_named_errors(call, match):
    """Each refused input raises with a message naming the problem."""
    with pytest.raises((ValueError, TypeError), match=match):
        call(make_system())


# --------------------------------------------------------------------------
# Positions and signs, not only "it moved"
# --------------------------------------------------------------------------


def _kernel_centroid(system, incl, pa):
    """Flux centroid (col, row offsets) of the kernel's rendered disk."""
    image = np.asarray(
        system.disk.surface_brightness(
            jnp.array(550.0), jnp.array(0.0), jnp.array(incl), jnp.array(pa)
        )
    )
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2)
    return np.array([(image * x).sum(), (image * y).sum()]) / image.sum()


@pytest.mark.parametrize(("incl", "pa"), [(60.0, 30.0), (40.0, 150.0)])
def test_image_sightline_marker_sits_on_the_projected_minor_axis(incl, pa):
    """The marker is r |cos i| / d arcsec out, on the bright side for +r."""
    system = make_system(incl, pa, disk=make_disk(g=0.6))
    centroid = _kernel_centroid(system, incl, pa)
    for radius in (3.0, -3.0):
        gids = by_gid(
            viz.plot_disk_image(system, wavelength_nm=550.0, grain_radius_AU=radius)
        )
        marker = gids["sightline"].get_offsets()[0]
        expected = abs(radius) * abs(np.cos(np.radians(incl))) / DIST_PC
        assert np.hypot(*marker) == pytest.approx(expected, rel=1e-9)
        # Perpendicular to the line of nodes, which lies at pa from +x.
        nodes = np.array([np.cos(np.radians(pa)), np.sin(np.radians(pa))])
        assert marker @ nodes == pytest.approx(0.0, abs=1e-12)
        # Near-side grains mark the forward-scattering (bright) half.
        assert np.sign(marker @ centroid) == np.sign(radius)


@pytest.mark.parametrize("dlon", [30.0, 135.0, 170.0])
def test_zodi_side_view_sun_is_at_r_cos_dlon(dlon):
    """The projected Sun sits at R cos(Delta lambda) on the ecliptic."""
    result = viz.plot_local_zodi_geometry(20.0, dlon, view="side")
    sun = by_gid(result)["sun"].get_offsets()[0]
    assert sun == pytest.approx([np.cos(np.radians(dlon)), 0.0])
    result.update(20.0, 60.0)
    sun = by_gid(result)["sun"].get_offsets()[0]
    assert sun == pytest.approx([np.cos(np.radians(60.0)), 0.0])


@pytest.mark.parametrize(("beta", "dlon"), [(0.0, 30.0), (25.0, 110.0), (-40.0, 160.0)])
def test_zodi_top_view_rays_point_the_right_way(beta, dlon):
    """Sunlight leaves the Sun toward the grain; scattered light heads home.

    Directions are compared as angles to 1e-4 degrees, the resolution
    arccos keeps near zero.

    The longitude arc starts on the Sun direction and ends on the
    sightline's projection, which leaves the observer toward +y.
    """
    gids = by_gid(viz.plot_local_zodi_geometry(beta, dlon))
    observer = gids["observer"].get_offsets()[0]
    grain = gids["grain"].get_offsets()[0]

    incident = arrow_vector(gids["incident"])
    assert angle_between(incident, grain) == pytest.approx(0.0, abs=1e-4)
    assert np.linalg.norm(gids["incident"].xyann) < np.linalg.norm(grain)

    scattered = arrow_vector(gids["scattered"])
    assert angle_between(scattered, observer - grain) == pytest.approx(0.0, abs=1e-4)

    (x0, y0), (x1, y1) = gids["sightline"].get_xydata()
    assert (x0, y0) == pytest.approx(tuple(observer))
    assert y1 > y0 or dlon == 0.0

    arc = gids["look_angle"].get_xydata()
    first, last = arc[0] - observer, arc[-1] - observer
    assert angle_between(first, [-1.0, 0.0]) == pytest.approx(0.0, abs=1e-4)
    assert angle_between(last, [x1 - x0, y1 - y0]) == pytest.approx(0.0, abs=1e-4)


def test_zodi_side_inset_turns_sunlight_the_right_way():
    """The side-view inset keeps the incident ray on its projected side."""
    beta, dlon = 30.0, 135.0
    gids = by_gid(
        viz.plot_local_zodi_geometry(beta, dlon, view="side", grain_distance_AU=0.5)
    )
    look = np.array(
        [
            -np.cos(np.radians(beta)) * np.cos(np.radians(dlon)),
            np.cos(np.radians(beta)) * np.sin(np.radians(dlon)),
            np.sin(np.radians(beta)),
        ]
    )
    grain3 = np.array([1.0, 0.0, 0.0]) + 0.5 * look
    along = np.array([-np.cos(np.radians(dlon)), np.sin(np.radians(dlon)), 0.0])
    k_in = grain3 / np.linalg.norm(grain3)
    expected_in = np.array([k_in @ along, k_in[2]])
    out = -np.array([np.cos(np.radians(beta)), np.sin(np.radians(beta))])
    inset_in = arrow_vector(gids["inset/incident"])
    inset_out = arrow_vector(gids["inset/scattered"])
    expected_side = np.sign(out[0] * expected_in[1] - out[1] * expected_in[0])
    drawn_side = np.sign(inset_out[0] * inset_in[1] - inset_out[1] * inset_in[0])
    assert drawn_side == expected_side != 0.0


@pytest.mark.parametrize(
    ("beta", "dlon"), [(30.0, 135.0), (30.0, 40.0), (-30.0, 150.0)]
)
def test_zodi_side_inset_leaves_the_sun_visible(beta, dlon):
    """The default side-view inset never covers the projected Sun."""
    result = viz.plot_local_zodi_geometry(beta, dlon, view="side")
    result.fig.canvas.draw()
    sun = by_gid(result)["sun"].get_offsets()[0]
    sun_px = result.ax.transData.transform(sun)
    (inset,) = result.ax.child_axes
    assert not inset.get_window_extent().contains(*sun_px)


def test_zodi_pole_look_empties_the_longitude_arc():
    """At beta = 90 there is no longitude difference to draw in the top view."""
    result = viz.plot_local_zodi_geometry(30.0, 100.0)
    result.update(90.0, 100.0)
    gids = by_gid(result)
    assert len(gids["look_angle"].get_xydata()) == 0
    assert gids["look_angle/label"].get_text() == ""


def test_face_on_disk_has_no_near_side_label():
    """Within a few degrees of face-on no half is labeled near."""
    system = make_system(2.0, 30.0)
    assert "label/near_side" not in by_gid(viz.plot_system(system, 0.0))
    geometry = viz.plot_disk_geometry(system)
    assert by_gid(geometry)["label/near_side"].get_text() == ""
    geometry.update(40.0)
    assert by_gid(geometry)["label/near_side"].get_text() == "near side"


# --------------------------------------------------------------------------
# Inclination range and nonnegative surface brightness
# --------------------------------------------------------------------------


def test_disk_image_draws_the_same_midplane_above_90_degrees():
    """``(120, 30)`` and ``(60, 210)`` are one midplane and draw one image."""
    system = make_system(120.0, 30.0, disk=make_disk(g=0.6))
    above = viz.plot_disk_image(system, wavelength_nm=550.0)
    below = viz.plot_disk_image(
        system.disk, wavelength_nm=550.0, incl_deg=60.0, pa_deg=210.0
    )
    a = np.asarray(above.artists["image"].get_array())
    b = np.asarray(below.artists["image"].get_array())
    assert np.all(a >= 0.0)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12 * b.max())


def test_disk_image_refuses_blank_and_negative_arrays():
    """A map with no positive pixel, or real negatives, is not drawn blank."""
    with pytest.raises(ValueError, match="nonnegative"):
        viz.plot_disk_image(np.zeros((8, 8)), pixel_scale_arcsec=0.1)
    image = np.ones((8, 8))
    image[2, 3] = -0.5
    with pytest.raises(ValueError, match="nonnegative"):
        viz.plot_disk_image(image, pixel_scale_arcsec=0.1)
    result = viz.plot_disk_image(np.ones((8, 8)), pixel_scale_arcsec=0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        result.update(-np.ones((8, 8)))


def test_every_view_accepts_zero_to_180_and_refuses_beyond():
    """All disk views share one inclination range, [0, 180]."""
    for incl in (0.0, 135.0, 180.0):
        viz.plot_system(make_system(incl, 30.0), 0.0)
        viz.plot_system(make_system(incl, 30.0), 0.0, view="side")
        viz.plot_disk_geometry(make_system(incl, 30.0))
    for incl in (-5.0, 185.0):
        with pytest.raises(ValueError, match="180"):
            viz.plot_system(make_system(incl, 30.0), 0.0)
        with pytest.raises(ValueError, match="180"):
            viz.plot_disk_geometry(make_system(incl, 30.0))
        with pytest.raises(ValueError, match="180"):
            viz.plot_disk_image(
                np.ones((8, 8)), pixel_scale_arcsec=0.1, incl_deg=incl, pa_deg=0.0
            )


# --------------------------------------------------------------------------
# The scattering-angle view
# --------------------------------------------------------------------------


def _degrees_in(text):
    """The integer degrees printed in an angle value, such as 154."""
    return int(text.split("=")[1].split("$")[0])


@pytest.mark.parametrize("theta", [20.0, 90.0, 154.0])
def test_scattering_angle_known_answer(theta):
    """Theta runs from the forward continuation of k_in to k_out; alpha is 180 - Theta.

    The rays point along their propagation directions, the forward line
    continues ``k_in`` past the grain, the star sits behind the incident ray
    and the observer ahead of the scattered one.
    """
    k_in = np.array([1.0, 0.0])
    t = np.radians(theta)
    k_out = np.array([np.cos(t), np.sin(t)])
    gids = by_gid(viz.plot_scattering_angle(k_in, k_out))

    assert _degrees_in(gids["scattering_angle/value"].get_text()) == round(theta)
    assert _degrees_in(gids["illumination_angle/value"].get_text()) == round(
        180.0 - theta
    )
    assert angle_between(arrow_vector(gids["incident"]), k_in) == pytest.approx(
        0.0, abs=1e-4
    )
    assert angle_between(arrow_vector(gids["scattered"]), k_out) == pytest.approx(
        0.0, abs=1e-4
    )
    (x0, y0), (x1, y1) = gids["forward"].get_xydata()
    assert angle_between([x1 - x0, y1 - y0], k_in) == pytest.approx(0.0, abs=1e-4)
    grain = gids["grain"].get_offsets()[0]
    assert angle_between(gids["star"].get_offsets()[0] - grain, -k_in) == (
        pytest.approx(0.0, abs=1e-4)
    )
    assert angle_between(gids["observer"].get_offsets()[0] - grain, k_out) == (
        pytest.approx(0.0, abs=1e-4)
    )
    for name in ("incident", "scattered", "forward"):
        assert gids[f"label/{name}"].get_text()


@pytest.mark.parametrize(
    ("beta", "dlon", "d"), [(30.0, 135.0, 1.0), (-20.0, 60.0, 0.4)]
)
def test_scattering_angle_draws_the_zodi_inset_construction(beta, dlon, d):
    """A 3D pair is turned exactly as the top view turns its grain inset.

    The grain ``d`` AU out along the look direction receives sunlight along
    the Sun-to-grain direction and sends light home along ``-look``; the
    full-size drawing and the top view's inset draw the same rays and print
    the same angles.
    """
    look = np.array(
        [
            -np.cos(np.radians(beta)) * np.cos(np.radians(dlon)),
            np.cos(np.radians(beta)) * np.sin(np.radians(dlon)),
            np.sin(np.radians(beta)),
        ]
    )
    grain = np.array([1.0, 0.0, 0.0]) + d * look
    full = by_gid(viz.plot_scattering_angle(grain, -look))
    inset = by_gid(viz.plot_local_zodi_geometry(beta, dlon, grain_distance_AU=d))
    for name in ("incident", "scattered"):
        np.testing.assert_allclose(
            arrow_vector(full[name]), arrow_vector(inset[f"inset/{name}"]), atol=1e-12
        )
    for name in ("scattering_angle/value", "illumination_angle/value"):
        assert full[name].get_text() == inset[f"inset/{name}"].get_text()


def test_scattering_angle_update_redraws_in_place():
    """``update`` moves the rays and reprints the angles, adding no artist."""
    result = viz.plot_scattering_angle([1.0, 0.0], [0.0, 1.0])
    n_children = len(result.ax.get_children())
    result.update([1.0, 0.0], [-1.0, 0.1])
    gids = by_gid(result)
    assert _degrees_in(gids["scattering_angle/value"].get_text()) == 174
    assert len(result.ax.get_children()) == n_children


def test_scattering_angle_optional_parts_are_absent_when_off():
    """Without star, observer and labels, only the construction is drawn."""
    gids = by_gid(
        viz.plot_scattering_angle(
            [1.0, 0.0], [0.0, 1.0], star=False, observer=False, labels=False
        )
    )
    assert not {"star", "observer", "label/incident"} & set(gids)
    assert {"incident", "scattered", "forward", "grain"} <= set(gids)


@pytest.mark.parametrize(
    ("k_in", "k_out"),
    [([1.0, 0.0], [0.0, 1.0, 0.0]), ([0.0, 0.0], [1.0, 0.0]), ([1.0], [1.0])],
)
def test_scattering_angle_refuses_bad_directions(k_in, k_out):
    """Mixed, zero or one-element directions raise a named error."""
    with pytest.raises(ValueError, match="k_in"):
        viz.plot_scattering_angle(k_in, k_out)


# --------------------------------------------------------------------------
# show: drawing a subset of a geometry view
# --------------------------------------------------------------------------

# Each part and the gids it owns, written out here rather than read from
# the views, so the tests pin the documented names.
ZODI_TOP_PARTS = {
    "orbit": ["observer_orbit"],
    "sun": ["sun", "label/sun", "to_sun"],
    "observer_mark": ["observer"],
    "observer_label": ["label/observer"],
    "sightline": ["sightline", "label/sightline"],
    "grain": ["grain"],
    "incident": ["incident"],
    "scattered": ["scattered"],
    "look_angle": ["look_angle", "look_angle/label"],
    "readout": ["label/angles"],
    "inset": [],
}
ZODI_SIDE_PARTS = {
    **{k: v for k, v in ZODI_TOP_PARTS.items() if k not in ("incident", "scattered")},
    "orbit": ["ecliptic"],
    "incident": [],
    "scattered": [],
}
DISK_PARTS = {
    "star": ["star"],
    "sky_plane": ["sky_plane", "label/sky_plane"],
    "disk": ["disk/near", "disk/far", "label/near_side", "label/far_side"],
    "layer": ["disk/layer_near", "disk/layer_far"],
    "sightline": ["sightline"],
    "grain": ["grain"],
    "incident": ["incident"],
    "scattered": ["scattered"],
    "scattering_angle": ["forward", "scattering_angle", "scattering_angle/label"],
    "inclination": ["inclination", "inclination/label"],
    "observer_mark": ["observer"],
    "observer_label": ["label/observer"],
    "inset": [],
}
DISK_IMAGE_PARTS = {
    "star": ["star"],
    "outline": ["disk/inner", "disk/outer"],
    "sightline": ["sightline"],
}
SHOW_VIEWS = {
    "zodi_top": (
        lambda show: viz.plot_local_zodi_geometry(30.0, 135.0, show=show),
        ZODI_TOP_PARTS,
    ),
    "zodi_side": (
        lambda show: viz.plot_local_zodi_geometry(30.0, 135.0, view="side", show=show),
        ZODI_SIDE_PARTS,
    ),
    "disk": (
        lambda show: viz.plot_disk_geometry(
            make_system(60.0, 0.0), thickness_AU=0.5, show=show
        ),
        DISK_PARTS,
    ),
    "disk_image": (
        lambda show: viz.plot_disk_image(
            make_system(60.0, 30.0), wavelength_nm=550.0, grain_radius_AU=3.0, show=show
        ),
        DISK_IMAGE_PARTS,
    ),
}


def _visible(result):
    """Visibility of every gid-carrying artist, plus the inset axes."""
    out = {gid: a.get_visible() for gid, a in by_gid(result).items()}
    for child in result.ax.child_axes:
        out[child.get_gid()] = child.get_visible()
    return out


@pytest.mark.parametrize("name", sorted(SHOW_VIEWS))
def test_show_none_draws_every_part(name):
    """The default draws every artist, as before ``show`` existed."""
    make, _ = SHOW_VIEWS[name]
    assert all(_visible(make(None)).values())


@pytest.mark.parametrize("name", sorted(SHOW_VIEWS))
def test_show_leaves_out_exactly_the_named_part(name):
    """Dropping one part hides its artists (and labels) and nothing else."""
    make, parts = SHOW_VIEWS[name]
    for part, gids in parts.items():
        visible = _visible(make([p for p in parts if p != part]))
        hidden = set(gids) | ({"inset"} if part == "inset" else set())
        if part in ("sightline", "layer") and "sightline/path" in visible:
            hidden.add("sightline/path")
        for gid, shown in visible.items():
            assert shown == (gid not in hidden), (part, gid)


@pytest.mark.parametrize("name", sorted(SHOW_VIEWS))
def test_show_draws_the_same_pixels_as_the_default_when_complete(name):
    """Naming every part renders byte for byte what the default renders."""
    import io

    make, parts = SHOW_VIEWS[name]
    renders = []
    for show in (None, list(parts)):
        result = make(show)
        buf = io.BytesIO()
        result.fig.savefig(buf, format="png")
        renders.append(buf.getvalue())
    assert renders[0] == renders[1]


def test_show_hidden_parts_still_move_with_update():
    """A hidden part keeps its gid and follows ``update``, ready to reveal."""
    result = viz.plot_disk_geometry(make_system(60.0, 0.0), show=("star", "disk"))
    before = state(by_gid(result)["grain"])
    result.update(30.0)
    grain = by_gid(result)["grain"]
    assert not grain.get_visible()
    assert not same(before, state(grain))


@pytest.mark.parametrize("name", ["zodi_top", "zodi_side", "disk"])
def test_show_observer_still_names_the_mark_and_its_label(name):
    """The name ``observer`` from before the split shows both halves."""
    make, parts = SHOW_VIEWS[name]
    rest = [p for p in parts if not p.startswith("observer_")]
    together = _visible(make([*rest, "observer"]))
    assert together == _visible(make(None))
    label_only = _visible(make([*rest, "observer_label"]))
    assert label_only["label/observer"]
    assert not label_only["observer"]


def test_observer_label_text_is_settable():
    """A figure can rename the observer, in both geometry views."""
    zodi = by_gid(viz.plot_local_zodi_geometry(30.0, 135.0, observer_text="telescope"))
    assert zodi["label/observer"].get_text() == "telescope"
    disk = by_gid(viz.plot_disk_geometry(make_system(), observer_text="telescope"))
    assert disk["label/observer"].get_text() == "telescope"


def test_show_accepts_one_name_and_refuses_unknown_parts():
    """A single part name is a one-part show; an unknown name is refused."""
    visible = _visible(viz.plot_local_zodi_geometry(30.0, 135.0, show="grain"))
    assert visible["grain"]
    assert not visible["sightline"]
    with pytest.raises(ValueError, match="look_arc"):
        viz.plot_local_zodi_geometry(30.0, 135.0, show=("grain", "look_arc"))
    with pytest.raises(ValueError, match="cloud"):
        viz.plot_disk_geometry(make_system(), show=("cloud",))
    with pytest.raises(ValueError, match="rings"):
        viz.plot_disk_image(
            np.ones((8, 8)), pixel_scale_arcsec=0.1, show=("star", "rings")
        )


# --------------------------------------------------------------------------
# Several grains on the local-zodi sightline
# --------------------------------------------------------------------------


def test_zodi_grains_each_lit_from_the_sun_at_their_own_distance():
    """Every grain sits on the sightline and gets its own sunlight ray.

    The first grain is the marked one: the inset prints its angles, the
    in-ecliptic 135 degrees of a grain 1 AU out at a 90 degree look.
    """
    distances = [1.0, 0.5, 1.5]
    result = viz.plot_local_zodi_geometry(0.0, 90.0, grain_distance_AU=distances)
    gids = by_gid(result)
    grains = gids["grain"].get_offsets()
    observer = np.array([1.0, 0.0])
    np.testing.assert_allclose(
        grains, observer + np.outer(distances, [0.0, 1.0]), atol=1e-12
    )
    for k, grain in enumerate(grains):
        tag = "" if k == 0 else f"/{k}"
        assert angle_between(arrow_vector(gids["incident" + tag]), grain) == (
            pytest.approx(0.0, abs=1e-4)
        )
        assert angle_between(
            arrow_vector(gids["scattered" + tag]), observer - grain
        ) == pytest.approx(0.0, abs=1e-4)
    assert "135" in gids["inset/scattering_angle/value"].get_text()

    before = state(gids["incident/2"])
    result.update(20.0, 60.0)
    assert not same(before, state(gids["incident/2"]))

    side = by_gid(
        viz.plot_local_zodi_geometry(
            30.0, 90.0, view="side", grain_distance_AU=distances
        )
    )
    radial = np.linalg.norm(side["grain"].get_offsets(), axis=1)
    np.testing.assert_allclose(radial, distances, atol=1e-12)


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]]])
def test_zodi_refuses_empty_or_nested_grain_distances(bad):
    """A grain distance is a number or a flat, nonempty sequence."""
    with pytest.raises(ValueError, match="grain_distance_AU"):
        viz.plot_local_zodi_geometry(30.0, 135.0, grain_distance_AU=bad)


# --------------------------------------------------------------------------
# Labels stay inside the figure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("font", ["Inter", "DejaVu Sans"])
@pytest.mark.parametrize("figsize", [(6.4, 4.8), (10.0, 4.5)])
def test_zodi_side_view_axis_label_fits_the_figure(figsize, font):
    """The side view's x label is not cut off at the figure edge.

    The wide case puts the side view in the right half of a two-panel
    figure, beside the top view, where the label used to run off the edge.
    Both are drawn in the house style, whose type is wider than the
    matplotlib default, and in DejaVu Sans, the wider fallback used where
    the house font is not installed (as on CI runners).
    """
    import hwostyle

    with hwostyle.light():
        plt.rcParams["font.family"] = [font]
        fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        viz.plot_local_zodi_geometry(30.0, 135.0, ax=axes[0])
        viz.plot_local_zodi_geometry(30.0, 135.0, view="side", ax=axes[1])
        fig.canvas.draw()
    box = axes[1].xaxis.label.get_window_extent()
    fig_box = fig.bbox
    assert box.x0 >= fig_box.x0 and box.x1 <= fig_box.x1
    assert box.y0 >= fig_box.y0


# --------------------------------------------------------------------------
# The grain drawing's options: gid prefix, where the values go, arc radii
# --------------------------------------------------------------------------


def test_scattering_angle_gid_prefix_matches_the_insets():
    """With ``gid_prefix="inset"`` the drawing carries the insets' gids."""
    look = np.array([0.6124, 0.6124, 0.5])
    grain = np.array([1.0, 0.0, 0.0]) + 0.5 * look
    drawing = by_gid(
        viz.plot_scattering_angle(
            grain, -look, star=False, observer=False, labels=False, gid_prefix="inset"
        )
    )
    view = by_gid(viz.plot_local_zodi_geometry(30.0, 135.0))
    assert set(drawing) == {g for g in view if g.startswith("inset/")}
    plain = by_gid(viz.plot_scattering_angle(grain, -look))
    assert not any(g.startswith("inset/") for g in plain)


@pytest.mark.parametrize("values", ["corner", "arcs", None])
def test_scattering_angle_values_go_where_asked(values):
    """The values print in the corners, on the arcs, or nowhere, and stay there."""
    result = viz.plot_scattering_angle([1.0, 0.0], [0.5, 0.8], values=values)
    for k_out in ([0.5, 0.8], [-1.0, 0.1]):
        result.update([1.0, 0.0], k_out)
        theta = round(float(np.degrees(np.arctan2(k_out[1], k_out[0]))))
        gids = by_gid(result)
        corner = [gids[f"{a}/value"].get_text() for a in ANGLES]
        arcs = [gids[f"{a}/label"].get_text() for a in ANGLES]
        if values == "corner":
            assert [_degrees_in(t) for t in corner] == [theta, 180 - theta]
            assert arcs == [r"$\Theta$", r"$\alpha$"]
        elif values == "arcs":
            assert corner == ["", ""]
            assert [_degrees_in(t) for t in arcs] == [theta, 180 - theta]
        else:
            assert corner == ["", ""]
            assert arcs == [r"$\Theta$", r"$\alpha$"]


ANGLES = ("scattering_angle", "illumination_angle")


@pytest.mark.parametrize(("theta_r", "alpha_r"), [(0.5, 0.3), (0.35, 0.6)])
def test_scattering_angle_arc_radii_are_settable(theta_r, alpha_r):
    """Each arc lies at its radius about the grain, its label beyond it."""
    gids = by_gid(
        viz.plot_scattering_angle(
            [1.0, 0.2], [0.3, 1.0], theta_radius=theta_r, alpha_radius=alpha_r
        )
    )
    for name, radius in zip(ANGLES, (theta_r, alpha_r), strict=True):
        r = np.linalg.norm(gids[name].get_xydata(), axis=1)
        np.testing.assert_allclose(r, radius, atol=1e-12)
        assert np.linalg.norm(gids[f"{name}/label"].get_position()) > radius


@pytest.mark.parametrize(
    "kwargs",
    [{"values": "legend"}, {"theta_radius": 0.0}, {"alpha_radius": -0.2}],
)
def test_scattering_angle_refuses_bad_options(kwargs):
    """An unknown value place or a non-positive radius is a named error."""
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        viz.plot_scattering_angle([1.0, 0.0], [0.0, 1.0], **kwargs)


def test_disk_inset_values_go_on_the_arcs():
    """``inset_values="arcs"`` moves the inset's values onto its arcs.

    The near-side minor-axis grain at 60 degrees scatters at 30 degrees.
    """
    result = viz.plot_disk_geometry(make_system(60.0, 0.0), inset_values="arcs")
    for incl in (60.0, 40.0):
        result.update(incl)
        gids = by_gid(result)
        theta = [_degrees_in(gids[f"inset/{a}/label"].get_text()) for a in ANGLES]
        assert theta == [round(90.0 - incl), round(90.0 + incl)]
        assert gids["inset/scattering_angle/value"].get_text() == ""
    with pytest.raises(ValueError, match="inset_values"):
        viz.plot_disk_geometry(make_system(), inset_values="legend")


# --------------------------------------------------------------------------
# The look-angle annotation: text, radius and label position
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("view", "value"), [("top", 135), ("side", 30)])
def test_look_angle_label_text_and_radius_are_settable(view, value):
    """The label is the template given the view's angle; the arc has the radius."""
    result = viz.plot_local_zodi_geometry(
        30.0,
        135.0,
        view=view,
        look_angle_label="angle\n= {value:.0f} deg",
        look_angle_radius_AU=0.45,
    )
    gids = by_gid(result)
    assert gids["look_angle/label"].get_text() == f"angle\n= {value} deg"
    vertex = np.array([1.0, 0.0]) if view == "top" else np.zeros(2)
    r = np.linalg.norm(gids["look_angle"].get_xydata() - vertex, axis=1)
    np.testing.assert_allclose(r, 0.45, atol=1e-12)
    result.update(20.0, 100.0)
    new = 100 if view == "top" else 20
    assert gids["look_angle/label"].get_text() == f"angle\n= {new} deg"


def test_look_angle_label_can_be_pinned():
    """A pinned label sits at its polar position, grows outward, and stays."""
    result = viz.plot_local_zodi_geometry(
        30.0, 135.0, look_angle_label_at=(100.0, 0.67)
    )
    label = by_gid(result)["look_angle/label"]
    at = np.radians(100.0)
    expected = np.array([1.0, 0.0]) + 0.67 * np.array([np.cos(at), np.sin(at)])
    for _ in range(2):
        np.testing.assert_allclose(label.get_position(), expected, atol=1e-12)
        assert (label.get_ha(), label.get_va()) == ("center", "bottom")
        result.update(10.0, 60.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"look_angle_radius_AU": 0.0},
        {"look_angle_label_at": (100.0, -0.2)},
        {"look_angle_label_at": 100.0},
    ],
)
def test_look_angle_refuses_bad_options(kwargs):
    """A non-positive radius or a malformed label position is a named error."""
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        viz.plot_local_zodi_geometry(30.0, 135.0, **kwargs)


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_degree_signs_sit_tight_against_their_numbers(name):
    """Every printed angle reads like 135 followed by the sign, one style.

    The sign is the mathtext degree glyph directly after the digits, never
    a raised circle (a mathtext superscript), which sets off after a gap.
    """
    from matplotlib.text import Text

    result = VIEWS[name](make_system(), None)
    texts = [t.get_text() for t in result.fig.findobj(Text) if t.get_text()]
    angles = [t for t in texts if "degree" in t or "circ" in t]
    if name.startswith(("zodi", "disk_geometry", "scattering")):
        assert angles, "the view prints at least one angle"
    for text in angles:
        assert "^\\circ" not in text, text
        assert re.search(r"\d\$\\degree\$", text), text


def _segment_enters(ax, start, end, box):
    """Whether the data-space segment passes through a display-space box."""
    points = np.linspace(np.asarray(start, float), np.asarray(end, float), 200)
    x, y = ax.transData.transform(points).T
    x0, y0, x1, y1 = box
    return bool(np.any((x > x0) & (x < x1) & (y > y0) & (y < y1)))


@pytest.mark.parametrize("font", ["Inter", "DejaVu Sans"])
def test_zodi_top_look_label_clears_the_rays(font):
    """The longitude label's box crosses no ray, line or arc of the top view.

    Checked over looks from narrow to wide, in the house font and in DejaVu
    Sans, the wider fallback used where the house font is not installed (as
    on CI runners), with a few points of margin, the backing box a label
    carries in a figure that boxes its text.
    """
    import hwostyle

    with hwostyle.light():
        plt.rcParams["font.family"] = [font]
        fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
        result = viz.plot_local_zodi_geometry(30.0, 135.0, ax=ax)
        gids = by_gid(result)
        label = gids["look_angle/label"]
        pad = 3.0 * fig.dpi / 72.0
        for beta in (-30.0, 0.0, 30.0):
            for dlon in (20.0, 45.0, 60.0, 75.0, 90.0, 120.0, 135.0, 160.0, 180.0):
                result.update(beta, dlon)
                fig.canvas.draw()
                bb = label.get_window_extent()
                box = (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                segments = {"sunlight": (gids["incident"].xyann, gids["incident"].xy)}
                for name in ("sightline", "look_angle", "to_sun"):
                    xy = gids[name].get_xydata()
                    for k in range(len(xy) - 1):
                        segments[f"{name}/{k}"] = (xy[k], xy[k + 1])
                crossed = [
                    name
                    for name, (a, b) in segments.items()
                    if _segment_enters(ax, a, b, box)
                ]
                assert not crossed, (beta, dlon, crossed)
                frame = ax.get_window_extent()
                assert frame.x0 <= bb.x0 and bb.x1 <= frame.x1, (beta, dlon)
                assert frame.y0 <= bb.y0 and bb.y1 <= frame.y1, (beta, dlon)

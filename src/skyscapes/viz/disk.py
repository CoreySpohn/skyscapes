"""Disk views: the projected surface-brightness map and the geometry behind it.

``plot_disk_radiance`` evaluates a disk's ``photon_radiance`` once and draws
it with ``eyepiece.plot.imshow_log`` on an arcsec grid, RA offset increasing to
the left. ``plot_disk_geometry`` draws the same disk edge-on to the reader:
its midplane at the inclination, the observer's direction, one sightline
through one grain, and the scattering angle at that grain. The two are
built to sit side by side: the sightline marked on the image is the one
drawn in the geometry panel.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import au_to_arcsec
from optixstuff.grids import Image, PixelGrid, SkyOrientation, resample

from skyscapes.disk import AbstractDisk
from skyscapes.disk.composite import orient_disk
from skyscapes.viz import _geometry
from skyscapes.viz._draw import (
    DEGREE,
    VALUE_PLACES,
    AngleArc,
    Arrow,
    GrainInset,
    halo,
    hide,
    resolve_show,
)
from skyscapes.viz._require import eyepiece

_SKY_LABELS = ("RA offset [arcsec]", "Dec offset [arcsec]")
# Below this sin(i) (about 6 degrees from face-on) no half of the disk is
# meaningfully nearer the observer, and the near/far labels are left off.
_MIN_SIN_FOR_NEAR = 0.1
GEOMETRY_PARTS = (
    "star",
    "sky_plane",
    "disk",
    "layer",
    "sightline",
    "grain",
    "incident",
    "scattered",
    "scattering_angle",
    "inclination",
    "observer_mark",
    "observer_label",
    "inset",
)
# Show names that stand for several parts: the observer's arrow and its
# label, the single part they were before they could be shown apart.
GEOMETRY_GROUPS = {"observer": ("observer_mark", "observer_label")}
IMAGE_PARTS = ("star", "outline", "sightline")
_SIDE_LABELS = (
    r"$z$, away from the observer [AU]",
    "offset along the projected minor axis [AU]",
)


def _resolve_disk(obj, incl_deg, position_angle_deg, dist_pc):
    """``(disk, incl, pa, dist)`` from a Scene, a disk, or neither."""
    system = obj
    if hasattr(system, "midplane_inc_deg") and hasattr(system, "disk"):
        incl = system.midplane_inc_deg if incl_deg is None else incl_deg
        pa = (
            system.midplane_position_angle_deg
            if position_angle_deg is None
            else position_angle_deg
        )
        dist = system.dist_pc if dist_pc is None else dist_pc
        return system.disk, incl, pa, dist
    if isinstance(obj, AbstractDisk):
        return obj, incl_deg, position_angle_deg, dist_pc
    return None, incl_deg, position_angle_deg, dist_pc


def plot_disk_radiance(
    disk_or_image,
    *,
    grid=None,
    star=None,
    wavelength_nm=None,
    time_d=0.0,
    incl_deg=None,
    position_angle_deg=None,
    pixel_scale_arcsec=None,
    dist_pc=None,
    radii_AU=None,
    outline=True,
    grain_radius_AU=None,
    dynamic_range=1.0e4,
    vmax=None,
    colorbar=True,
    show=None,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a disk's projected surface brightness, log-scaled, in arcsec.

    Disk rendering requires an explicit grid and host star (a system supplies
    the star and distance). Image inputs carry their sky geometry; bare
    arrays are north-up photon radiance and require pixel_scale_arcsec.

    Args:
        disk_or_image: A ``Scene`` (its disk, midplane
            orientation and distance are used), an ``AbstractDisk``, or a
            bare ``(ny, nx)`` photon radiance map.
        grid: Arcsec PixelGrid with SkyOrientation, required for rendering.
        star: Host star; a system supplies it.
        wavelength_nm: Wavelength to render at [nm]. Required for a disk.
        time_d: Time to render at, in days from the scene epoch.
        incl_deg: Midplane inclination [deg], in ``[0, 180]``. Defaults to
            the system's; required for a bare disk; optional for an array
            (outline only). ``i`` and ``180 - i`` at ``pa + 180`` are the
            same midplane and render the same map.
        position_angle_deg: Major-axis angle north through east [deg].
        pixel_scale_arcsec: Sampling for a bare north-up radiance array.
        dist_pc: Distance [pc], used to place the AU outlines. Read from
            the system; required for a bare parametric disk.
        radii_AU: ``(r_in, r_out)`` for the outlines. None reads a
            parametric disk's truncation radii.
        outline: Whether to draw the projected truncation rings.
        grain_radius_AU: Optional disk radius of the grain whose sightline
            ``plot_disk_geometry`` draws (negative for the far half); its
            sky position is marked.
        dynamic_range: Ratio of the colormap's top to its bottom. The
            bottom is the top over this, and fainter pixels (the inner
            hole) render at the floor color.
        vmax: Top of the colormap. None takes the image peak; pin it to
            the brightest frame when the result will be updated.
        colorbar: Forwarded to ``eyepiece.plot.imshow_log``.
        show: The overlays to draw over the image, as in the geometry
            views; None draws every one the inputs allow. Part names:
            ``"star"``, ``"outline"`` (the truncation rings) and
            ``"sightline"`` (the grain's marker). A part left out is
            created hidden (``set_visible(False)``), still in the result
            and still moved by ``update``, so a row of panels can share
            one set of gids and drop a mark where the comparison does not
            need it. The image and its colorbar are always drawn.
        ax: Axes to draw into. None creates a new figure and axes.
        imshow_kw: Extra kwargs for ``ax.imshow``.
        cbar_kw: Extra kwargs for the colorbar.

    Returns:
        An ``eyepiece.PlotResult`` with ``"image"`` (and ``"cbar"``),
        ``"lines"`` (gids ``disk/inner``, ``disk/outer``) when outlined, and
        ``"scatter"`` (``star``, and ``sightline`` when marked).
        ``update(image, incl_deg=None, position_angle_deg=None)`` sets new image data
        under the first draw's norm and, when angles are given, moves the
        outlines and the sightline marker; it computes no surface
        brightness, so render the frames first.

    Raises:
        ValueError: If a required input is missing, the image is not 2D,
            the inclination is outside ``[0, 180]``, a ``show`` part is
            unknown, or the map has no positive pixel or has negative
            pixels beyond round-off (at ``1e-6`` of the largest
            magnitude). A surface brightness cannot be negative.
            ``update`` applies the same check to each new frame.
    """
    ep = eyepiece()
    shown = resolve_show(show, IMAGE_PARTS)
    disk, incl, pa, dist = _resolve_disk(
        disk_or_image, incl_deg, position_angle_deg, dist_pc
    )
    if incl is not None:
        _check_incl(incl)
    if disk is not None:
        system = disk_or_image
        if star is None:
            star = getattr(system, "star", None)
        if wavelength_nm is None or star is None or grid is None or dist is None:
            raise ValueError("rendering needs wavelength_nm, star, dist_pc and grid")
        if incl is not None and pa is not None:
            disk = orient_disk(disk, incl, pa)
        image = disk.photon_radiance(
            jnp.asarray(wavelength_nm),
            jnp.asarray(time_d),
            star=star,
            dist_pc=dist,
            grid=grid,
        )
    elif hasattr(disk_or_image, "midplane_inc_deg"):
        raise ValueError("the scene has no disk to draw")
    else:
        image = disk_or_image
    display = None
    if isinstance(image, Image):
        grid = image.grid
        # Display in east/north while retaining the full input grid's footprint.
        ny, nx = grid.shape
        x, y = grid.pixel_to_unit(
            jnp.array([-0.5, nx - 0.5, -0.5, nx - 0.5]),
            jnp.array([-0.5, -0.5, ny - 0.5, ny - 0.5]),
        )
        east, north = grid.orientation.detector_to_sky(x, y)
        emin, emax, nmin, nmax = map(
            float, (east.min(), east.max(), north.min(), north.max())
        )
        dx, dy = (emax - emin) / nx, (nmax - nmin) / ny
        display = PixelGrid(
            (ny, nx),
            emax / dx - 0.5,
            -nmin / dy - 0.5,
            ((dx, 0.0), (0.0, dy)),
            "arcsec",
            orientation=SkyOrientation(0.0),
        )
        image = resample(image, display).data
        extent = (emax, emin, nmin, nmax)
        pixel_scale_arcsec = dx
    else:
        extent = None
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    _check_brightness(image)
    if pixel_scale_arcsec is None:
        raise ValueError("pixel_scale_arcsec is required for a bare array")

    ny, nx = image.shape
    half_x = 0.5 * nx * float(pixel_scale_arcsec)
    half_y = 0.5 * ny * float(pixel_scale_arcsec)
    peak = float(np.nanmax(image)) if vmax is None else float(vmax)
    floor = peak / float(dynamic_range) if peak > 0.0 else 1e-30
    base = ep.plot.imshow_log(
        image,
        ax=ax,
        extent=extent if extent is not None else (half_x, -half_x, -half_y, half_y),
        floor=floor,
        vmin=floor,
        vmax=max(peak, 10.0 * floor),
        colorbar=colorbar,
        cbar_label="disk radiance [ph s^-1 m^-2 nm^-1 sr^-1]",
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )
    ax = base.ax
    ax.set_ylim(*(extent[2:] if extent is not None else (-half_y, half_y)))
    ax.set_xlabel(_SKY_LABELS[0])
    ax.set_ylabel(_SKY_LABELS[1])

    artists = dict(base.artists)
    star = ax.scatter(
        [0.0], [0.0], s=90, marker="*", color=eyepiece().style.role("star"), zorder=5
    )
    star.set_gid("star")
    scatters = [star]

    radii = radii_AU
    if radii is None and disk is not None:
        radii = _geometry.disk_radii_AU(disk)
    can_outline = outline and radii is not None and dist is not None
    can_outline = can_outline and incl is not None and pa is not None
    rings = []
    if can_outline:
        for name in ("inner", "outer"):
            (line,) = ax.plot(
                [], [], color=eyepiece().style.role("disk"), lw=0.9, ls="--", alpha=0.9
            )
            line.set_gid(f"disk/{name}")
            rings.append(line)
        artists["lines"] = rings

    marker = None
    if grain_radius_AU is not None and dist is not None and incl is not None:
        marker = ax.scatter(
            [0.0],
            [0.0],
            s=70,
            facecolors="none",
            edgecolors=eyepiece().style.neutral(0.6),
            linewidths=1.4,
            zorder=6,
        )
        marker.set_gid("sightline")
        scatters.append(marker)
    artists["scatter"] = scatters

    def place(incl_now, pa_now):
        """Move the outlines and the sightline marker to an orientation.

        Args:
            incl_now: Midplane inclination [deg].
            pa_now: Midplane position angle [deg].
        """
        if rings:
            for line, radius in zip(rings, radii, strict=True):
                ring = _geometry.ring_sky(radius, incl_now, pa_now)
                line.set_data(
                    au_to_arcsec(ring[:, 0], dist), au_to_arcsec(ring[:, 1], dist)
                )
        if marker is not None:
            _, near, _ = _geometry.disk_axes_sky(incl_now, pa_now)
            grain = float(grain_radius_AU) * near
            marker.set_offsets(
                [[au_to_arcsec(grain[0], dist), au_to_arcsec(grain[1], dist)]]
            )

    state = {"incl": incl, "pa": pa}
    if incl is not None and pa is not None:
        place(float(incl), float(pa))
    hide({"star": [star], "outline": rings, "sightline": [marker]}, shown)

    def update(new_image, incl_deg=None, position_angle_deg=None):
        """Show a new image; move the outlines when angles are given.

        Args:
            new_image: New map, an ``(ny, nx)`` photon radiance array or an
                ``Image`` that is resampled onto the first draw's display
                grid.
            incl_deg: New midplane inclination [deg]; None keeps the
                current one.
            position_angle_deg: New major-axis angle [deg]; None keeps the
                current one.

        Returns:
            None.
        """
        if isinstance(new_image, Image):
            target = (
                display
                if display is not None
                else PixelGrid.regular(
                    (ny, nx),
                    pixel_scale_arcsec,
                    "arcsec",
                    orientation=SkyOrientation(0.0),
                )
            )
            new_image = resample(new_image, target).data
        new_image = np.asarray(new_image, dtype=float)
        if incl_deg is not None:
            _check_incl(incl_deg)
        _check_brightness(new_image)
        base.update(new_image)
        if incl_deg is None and position_angle_deg is None:
            return
        state["incl"] = state["incl"] if incl_deg is None else incl_deg
        state["pa"] = state["pa"] if position_angle_deg is None else position_angle_deg
        if state["incl"] is not None and state["pa"] is not None:
            place(float(state["incl"]), float(state["pa"]))

    return ep.PlotResult(ax=ax, artists=artists, update=update)


class _GeometryPanel:
    """The artists of a disk geometry panel, placed for one inclination."""

    def __init__(
        self,
        ax,
        radii,
        pa,
        grain_radius,
        thickness,
        labels,
        inset,
        *,
        observer_text="to observer",
        inset_values="corner",
    ):
        """Create the panel's artists; place them with ``place``.

        Args:
            ax: The axes to draw on.
            radii: ``(r_in, r_out)`` truncation radii [AU].
            pa: Midplane position angle [deg].
            grain_radius: Disk-frame radius of the grain [AU]; negative
                for the far half.
            thickness: Full thickness of the dust layer [AU], or None for
                no layer.
            labels: Whether to draw the text labels.
            inset: ``(x0, y0, width, height)`` of the grain inset in axes
                coordinates, or None for no inset.
            observer_text: Text of the observer label.
            inset_values: Where the inset prints its angle values; see
                ``ScatteringAngle``.
        """
        self.ax = ax
        self.radii = radii
        self.pa = pa
        self.minor = _geometry.minor_axis_direction(pa)
        self.grain_radius = grain_radius
        self.thickness = thickness
        self.big = radii[1]
        disk_color = eyepiece().style.role("disk")
        scenery = eyepiece().style.neutral(0.45)
        R = self.big

        (self.plane,) = ax.plot([0.0, 0.0], [-1.2 * R, 1.2 * R], color=scenery)
        self.plane.set(lw=0.8, ls=":", gid="sky_plane")
        self.layers = []
        if thickness is not None:
            for name in ("near", "far"):
                (poly,) = ax.fill([0.0], [0.0], color=disk_color, alpha=0.25, lw=0)
                poly.set_gid(f"disk/layer_{name}")
                self.layers.append(poly)
        (self.near,) = ax.plot([], [], color=disk_color, lw=2.4)
        self.near.set_gid("disk/near")
        (self.far,) = ax.plot([], [], color=disk_color, lw=2.4)
        self.far.set_gid("disk/far")
        (self.sightline,) = ax.plot([], [], color=eyepiece().style.neutral(0.6), lw=0.9)
        self.sightline.set_gid("sightline")
        self.path = None
        if thickness is not None:
            (self.path,) = ax.plot([], [], color=disk_color, lw=4.0, alpha=0.8)
            self.path.set_gid("sightline/path")
        (self.forward,) = ax.plot([], [], color=scenery, lw=0.9, ls="--")
        self.forward.set_gid("forward")
        self.star = ax.scatter(
            [0.0],
            [0.0],
            s=170,
            marker="*",
            color=eyepiece().style.role("star"),
            zorder=5,
        )
        self.star.set_gid("star")
        self.grain = ax.scatter(
            [0.0], [0.0], s=40, color=disk_color, zorder=6, edgecolors="none"
        )
        self.grain.set_gid("grain")
        self.incident = Arrow(
            ax, (0, 0), (0, 0), color=eyepiece().style.role("star"), gid="incident"
        )
        self.scattered = Arrow(ax, (0, 0), (0, 0), color=disk_color, gid="scattered")
        self.observer = Arrow(
            ax,
            (-1.05 * R, 0.95 * R),
            (-1.4 * R, 0.95 * R),
            color=eyepiece().style.neutral(0.75),
            gid="observer",
        )
        self.theta = AngleArc(
            ax, color=eyepiece().style.role("text"), gid="scattering_angle"
        )
        self.incl_arc = AngleArc(
            ax, color=eyepiece().style.neutral(0.75), gid="inclination"
        )
        self.texts = [self.observer.artist, self.incident.artist]
        self.texts += [self.scattered.artist, self.theta.text, self.incl_arc.text]
        self.near_label = self.far_label = None
        self.observer_label = self.sky_label = None
        if labels:
            common = {"fontsize": "small", "ha": "center", "va": "center"}
            self.near_label = ax.text(0, 0, "near side", color=disk_color, **common)
            self.near_label.set_gid("label/near_side")
            self.far_label = ax.text(0, 0, "far side", color=disk_color, **common)
            self.far_label.set_gid("label/far_side")
            observer = ax.text(
                -1.22 * R,
                1.0 * R,
                observer_text,
                color=eyepiece().style.neutral(0.75),
                fontsize="small",
                ha="center",
                va="bottom",
            )
            observer.set_gid("label/observer")
            sky = ax.text(
                0.03 * R,
                -1.15 * R,
                "sky plane",
                color=scenery,
                fontsize="small",
                ha="left",
                va="bottom",
            )
            sky.set_gid("label/sky_plane")
            self.texts += [self.near_label, self.far_label, observer, sky]
            self.observer_label, self.sky_label = observer, sky
            for text in self.texts[-4:]:
                halo(text)
        self.inset = None
        if inset is not None:
            self.inset = GrainInset(
                ax, inset, grain_color=disk_color, values=inset_values
            )

    def parts(self, shown):
        """The panel's artists grouped by ``show`` part name.

        The sightline's chord through the layer belongs to both
        ``sightline`` and ``layer``: it is drawn only when both are.

        Args:
            shown: The set of part names that are shown.

        Returns:
            Dict from part name to a list of artists (entries may be None).
        """
        parts = {
            "star": [self.star],
            "sky_plane": [self.plane, self.sky_label],
            "disk": [self.near, self.far, self.near_label, self.far_label],
            "layer": list(self.layers),
            "sightline": [self.sightline],
            "grain": [self.grain],
            "incident": [self.incident.artist],
            "scattered": [self.scattered.artist],
            "scattering_angle": [self.forward, self.theta.line, self.theta.text],
            "inclination": [self.incl_arc.line, self.incl_arc.text],
            "observer_mark": [self.observer.artist],
            "observer_label": [self.observer_label],
            "inset": [None if self.inset is None else self.inset.ax],
        }
        for part in ("sightline", "layer"):
            if part not in shown:
                parts[part].append(self.path)
                break
        return parts

    def side(self, points):
        """Side-view ``(z, u)`` pairs of sky-frame points.

        Args:
            points: Sky-frame ``(east, north, away)`` points [AU], shape
                ``(..., 3)``.

        Returns:
            Array of shape ``(..., 2)`` with ``z`` (away from the observer)
            and ``u`` (along the projected minor axis), in AU.
        """
        z, u = _geometry.to_side(points, self.minor)
        return np.stack([z, u], axis=-1)

    def place(self, incl):
        """Put every inclination-dependent artist where ``incl`` puts it.

        Args:
            incl: Midplane inclination [deg].
        """
        r_in, r_out = self.radii
        R = self.big
        _, near, normal = _geometry.disk_axes_sky(incl, self.pa)
        d = self.side(near)
        nrm = self.side(normal)
        self.near.set_data(*np.stack([r_in * d, r_out * d]).T)
        self.far.set_data(*np.stack([-r_in * d, -r_out * d]).T)
        if self.layers:
            h = 0.5 * self.thickness
            for poly, sign in zip(self.layers, (1.0, -1.0), strict=True):
                a, b = sign * r_in * d, sign * r_out * d
                poly.set_xy(
                    np.stack([a + h * nrm, b + h * nrm, b - h * nrm, a - h * nrm])
                )

        grain3 = float(self.grain_radius) * near
        grain = self.side(grain3)
        self.grain.set_offsets([grain])
        self.sightline.set_data([-1.42 * R, 1.2 * R], [grain[1], grain[1]])
        if self.path is not None:
            # The sightline's chords through the dust support: the parts of
            # the horizontal line u = grain_u inside each layer rectangle
            # (disk-frame radius in [r_in, r_out], |height| <= h/2).
            zs, us = [], []
            for poly in self.layers:
                chord = _horizontal_chord(poly.get_xy()[:4], grain[1])
                if chord is not None:
                    zs += [chord[0], chord[1], np.nan]
                    us += [grain[1], grain[1], np.nan]
            self.path.set_data(zs, us)

        k_in = grain / np.linalg.norm(grain)
        start = 0.12 * R * k_in
        self.incident.set(start, grain - 0.05 * R * k_in)
        self.scattered.set(
            grain - np.array([0.04 * R, 0.0]), grain - np.array([0.42 * R, 0.0])
        )
        self.forward.set_data(*np.stack([grain, grain + 0.3 * R * k_in]).T)
        self.theta.set(grain, k_in, np.array([-1.0, 0.0]), 0.16 * R, r"$\Theta$", 1.9)
        self.incl_arc.set(
            (0.0, 0.0),
            np.array([0.0, 1.0]),
            d,
            0.32 * R,
            rf"$i$ = {float(incl):.0f}{DEGREE}",
            1.4,
        )
        if self.near_label is not None:
            # Face-on, neither half is nearer: the labels go blank.
            tilted = abs(np.sin(np.radians(incl))) >= _MIN_SIN_FOR_NEAR
            self.near_label.set_text("near side" if tilted else "")
            self.far_label.set_text("far side" if tilted else "")
            self.near_label.set_position(tuple(1.14 * r_out * d))
            self.far_label.set_position(tuple(-1.14 * r_out * d))
        if self.inset is not None:
            self.inset.set(k_in, np.array([-1.0, 0.0]))

    def artists(self):
        """The panel's artists, grouped by artist-vocabulary key.

        Returns:
            Dict with keys ``"scatter"``, ``"lines"``, ``"text"`` and, when a
            layer is drawn, ``"fill"``, each a list of artists.
        """
        lines = [self.plane, self.near, self.far, self.sightline, self.forward]
        lines += [self.theta.line, self.incl_arc.line]
        if self.path is not None:
            lines.append(self.path)
        out = {
            "scatter": [self.star, self.grain],
            "lines": lines,
            "text": list(self.texts),
        }
        if self.layers:
            out["fill"] = list(self.layers)
        if self.inset is not None:
            for key, values in self.inset.artists().items():
                out[key] = out[key] + values
        return out


def _horizontal_chord(corners, u):
    """The ``(z_min, z_max)`` of the line at height ``u`` inside a convex polygon.

    Args:
        corners: Polygon vertices ``(z, u)``, shape ``(n, 2)``, in order.
        u: Height of the horizontal line.

    Returns:
        The chord's end points, or None when the line misses the polygon.
    """
    crossings = []
    n = len(corners)
    for k in range(n):
        (z0, u0), (z1, u1) = corners[k], corners[(k + 1) % n]
        if (u0 - u) * (u1 - u) > 0.0 or u0 == u1:
            continue
        crossings.append(z0 + (u - u0) * (z1 - z0) / (u1 - u0))
    if not crossings:
        return None
    return min(crossings), max(crossings)


def plot_disk_geometry(
    system_or_disk,
    *,
    incl_deg=None,
    position_angle_deg=None,
    radii_AU=None,
    grain_radius_AU=None,
    thickness_AU=None,
    inset=True,
    inset_bounds=None,
    inset_values="corner",
    labels=True,
    observer_text="to observer",
    show=None,
    ax=None,
):
    """Draw a disk edge-on to the reader: midplane, observer and one sightline.

    The panel is the plane holding the line of sight (horizontal, observer
    to the left along ``-z``) and the disk's projected minor axis
    (vertical); the line of nodes points out of the page. The midplane is
    the line at the inclination from the sky plane, from the inner to the
    outer truncation radius on each side, with its near half on the
    observer's side, as the library's disk kernels place it.

    One grain sits in the midplane at ``grain_radius_AU`` along the minor
    axis. The panel draws the sightline through it, the starlight arriving
    at it, the light it scatters toward the observer, and the scattering
    angle between the two, which the disk kernels evaluate their phase
    functions at. For a grain on the minor axis and ``i <= 90`` that angle
    is ``90 - i`` on the near side (forward scattering) and ``90 + i`` on
    the far side. An inset magnifies the grain and sets the scattering angle
    beside its supplement, the illumination angle.

    Args:
        system_or_disk: A ``Scene`` (its disk and midplane
            orientation are used) or an ``AbstractDisk``.
        incl_deg: Midplane inclination [deg], in ``[0, 180]``; defaults to
            the system's. Above 90 the near half lies below the sky-plane
            crossing of the minor axis, and a near-side grain scatters at
            ``i - 90``.
        position_angle_deg: Major-axis angle north through east [deg].
        radii_AU: ``(r_in, r_out)``. None reads a parametric disk's
            truncation radii.
        grain_radius_AU: Disk radius of the marked grain, positive on the
            near half and negative on the far half. None takes the middle
            of the disk.
        thickness_AU: Optional full thickness of a schematic dust layer,
            drawn as a band about the midplane, with the sightline's chord
            through it highlighted. None draws the midplane only.
        inset: Whether to draw the grain inset.
        inset_bounds: The inset's ``[x0, y0, width, height]`` in axes
            fractions (``Axes.inset_axes``). None puts it in the quadrant
            the disk and the sightline leave empty: lower right for a
            near-side grain, upper left for a far-side one.
        inset_values: Where the inset prints the two angle values, as
            ``values`` of ``plot_scattering_angle``: ``"corner"``,
            ``"arcs"`` (on the arc labels, which a small inset has more
            room for) or None.
        labels: Whether to add direct text labels.
        observer_text: Text of the observer's label, for a figure that
            names the observer (a telescope, say) otherwise.
        show: The parts to draw, for a figure that builds the view up
            part by part; None draws them all. Part names (each takes its
            own labels with it): ``"star"``, ``"sky_plane"``, ``"disk"``
            (the midplane halves and their near and far labels),
            ``"layer"`` (the schematic layer bands), ``"sightline"``,
            ``"grain"``, ``"incident"``, ``"scattered"``,
            ``"scattering_angle"`` (the forward continuation and the
            arc), ``"inclination"`` (the arc), ``"observer_mark"`` (the
            arrow), ``"observer_label"`` and ``"inset"``;
            ``"observer"`` names the arrow and its label together. Showing
            the label without the arrow keeps the name where a figure
            draws its own observer. The sightline's chord through
            the layer is drawn only when both ``"sightline"`` and
            ``"layer"`` are. A part left out is created hidden
            (``set_visible(False)``), still in the result and still moved
            by ``update``, so the view keeps one set of gids however it is
            shown.
        ax: Axes to draw into. None creates a new figure and axes.

    Returns:
        An ``eyepiece.PlotResult`` with ``"scatter"``, ``"lines"``,
        ``"text"`` (annotations include the arrows) and, with a thickness,
        ``"fill"``, which holds the two layer bands as ``Polygon`` patches
        (``ax.fill``). The inset's artists are appended after the panel's
        own but live on the inset, a child axes of ``result.ax`` whose gid
        is ``inset`` (``artist.axes`` reaches it); their gids start with
        ``inset/``. Each
        artist carries a ``gid``: ``star``, ``grain``, ``sky_plane``,
        ``disk/near``, ``disk/far``, ``disk/layer_near``,
        ``disk/layer_far``, ``sightline``, ``sightline/path``,
        ``forward``, ``incident``, ``scattered``, ``observer``,
        ``scattering_angle``, ``inclination`` (each arc's text adds
        ``/label``), ``label/...``, and ``inset/...``.
        ``update(incl_deg)`` moves every inclination-dependent artist and
        leaves ``star``, ``sky_plane``, ``observer`` and the fixed labels
        untouched, for an inclination sweep.

    Raises:
        ValueError: If no radii are available, the inclination is
            outside ``[0, 180]``, a ``show`` part is unknown, or
            ``inset_values`` is not one of ``"corner"``, ``"arcs"``, None.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt

    if inset_values not in VALUE_PLACES:
        raise ValueError(
            f"inset_values must be one of {VALUE_PLACES}, got {inset_values!r}"
        )

    shown = resolve_show(show, GEOMETRY_PARTS, GEOMETRY_GROUPS)
    disk, incl, pa, _ = _resolve_disk(system_or_disk, incl_deg, position_angle_deg, 0.0)
    if incl is None:
        raise ValueError("incl_deg is required for a bare disk")
    pa = 0.0 if pa is None else float(pa)
    radii = radii_AU if radii_AU is not None else _geometry.disk_radii_AU(disk)
    if radii is None:
        raise ValueError("pass radii_AU: this disk declares no truncation radii")
    radii = (float(radii[0]), float(radii[1]))
    if grain_radius_AU is None:
        grain_radius_AU = 0.5 * (radii[0] + radii[1])

    if inset_bounds is None:
        near_grain = grain_radius_AU > 0.0
        inset_bounds = (
            (0.02, 0.02, 0.36, 0.36) if near_grain else (0.62, 0.62, 0.36, 0.36)
        )

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    panel = _GeometryPanel(
        ax,
        radii,
        pa,
        float(grain_radius_AU),
        thickness_AU,
        labels,
        inset_bounds if inset else None,
        observer_text=observer_text,
        inset_values=inset_values,
    )

    def update(incl_deg):
        """Move the midplane, sightline, rays and arcs to ``incl_deg``.

        Args:
            incl_deg: Midplane inclination [deg], in ``[0, 180]``.
        """
        _check_incl(incl_deg)
        panel.place(float(incl_deg))

    _check_incl(incl)
    panel.place(float(incl))
    hide(panel.parts(shown), shown)
    R = radii[1]
    ax.set_xlim(-1.45 * R, 1.2 * R)
    ax.set_ylim(-1.2 * R, 1.2 * R)
    ax.set_aspect("equal")
    ax.set_xlabel(_SIDE_LABELS[0])
    ax.set_ylabel(_SIDE_LABELS[1])
    return ep.PlotResult(ax=ax, artists=panel.artists(), update=update)


def _check_brightness(image):
    """Refuse a map a log display would silently blank or misrepresent."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("the disk map has no finite pixels")
    peak = float(finite.max())
    worst = float(finite.min())
    scale = float(np.max(np.abs(finite)))
    if peak > 0.0 and worst >= -1e-6 * scale:
        return
    raise ValueError(
        f"the disk map is not a nonnegative surface brightness (peak {peak:.3g},"
        f" minimum {worst:.3g}); a log display would draw it blank or"
        f" clipped."
    )


def _check_incl(incl_deg):
    """Reject inclinations outside the documented ``[0, 180]`` range."""
    if not 0.0 <= float(incl_deg) <= 180.0:
        raise ValueError(f"incl_deg must lie in [0, 180], got {incl_deg}")

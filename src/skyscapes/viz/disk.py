"""Disk views: the projected surface-brightness map and the geometry behind it.

``plot_disk_image`` evaluates a disk's ``surface_brightness`` once and draws
it with ``eyepiece.imshow_log`` on an arcsec grid, RA offset increasing to
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

from skyscapes.disk import AbstractDisk
from skyscapes.viz import _geometry, _style
from skyscapes.viz._draw import AngleArc, Arrow, GrainInset, halo
from skyscapes.viz._require import eyepiece

_SKY_LABELS = ("RA offset [arcsec]", "Dec offset [arcsec]")
_SIDE_LABELS = (
    r"$z$, toward the observer [AU]",
    "offset along the projected minor axis [AU]",
)


def _resolve_disk(obj, incl_deg, pa_deg, dist_pc):
    """``(disk, incl, pa, dist)`` from a System/Scene, a disk, or neither."""
    system = getattr(obj, "system", obj)
    if hasattr(system, "midplane_inc_deg") and hasattr(system, "disk"):
        incl = system.midplane_inc_deg if incl_deg is None else incl_deg
        pa = system.midplane_pa_deg if pa_deg is None else pa_deg
        dist = system.star.dist_pc if dist_pc is None else dist_pc
        return system.disk, incl, pa, dist
    if isinstance(obj, AbstractDisk):
        dist = getattr(obj, "dist_pc", None) if dist_pc is None else dist_pc
        return obj, incl_deg, pa_deg, dist
    return None, incl_deg, pa_deg, dist_pc


def plot_disk_image(
    disk_or_image,
    *,
    wavelength_nm=None,
    time_jd=0.0,
    incl_deg=None,
    pa_deg=None,
    pixel_scale_arcsec=None,
    dist_pc=None,
    radii_AU=None,
    outline=True,
    grain_radius_AU=None,
    dynamic_range=1.0e4,
    vmax=None,
    colorbar=True,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a disk's projected surface brightness, log-scaled, in arcsec.

    A disk is rendered once here with its own ``surface_brightness`` at
    the given wavelength, time, inclination and position angle; a bare
    ``(ny, nx)`` array is drawn as given. Pixel ``(row, col)`` sits at
    sky ``(y, x)``, with ``x`` the RA offset, drawn increasing to the left.
    The disk's truncation rings, projected at the same orientation, are
    overlaid as outlines, so the geometry and the rendered light can be
    checked against each other by eye.

    Args:
        disk_or_image: A ``System`` or ``Scene`` (its disk, midplane
            orientation and distance are used), an ``AbstractDisk``, or a
            bare ``(ny, nx)`` contrast map.
        wavelength_nm: Wavelength to render at [nm]. Required for a disk.
        time_jd: Time to render at [JD].
        incl_deg: Midplane inclination [deg]. Defaults to the system's;
            required for a bare disk; optional for an array (outline only).
        pa_deg: Midplane position angle [deg], as ``incl_deg``.
        pixel_scale_arcsec: Pixel scale [arcsec/pixel]. Read from the disk
            when it has one; required for an array.
        dist_pc: Distance [pc], used to place the AU outlines. Read from
            the system or disk when available.
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
        colorbar: Forwarded to ``eyepiece.imshow_log``.
        ax: Axes to draw into. None creates a new figure and axes.
        imshow_kw: Extra kwargs for ``ax.imshow``.
        cbar_kw: Extra kwargs for the colorbar.

    Returns:
        An ``eyepiece.PlotResult`` with ``"image"`` (and ``"cbar"``),
        ``"lines"`` (gids ``disk/inner``, ``disk/outer``) when outlined, and
        ``"scatter"`` (``star``, and ``sightline`` when marked).
        ``update(image, incl_deg=None, pa_deg=None)`` sets new image data
        under the first draw's norm and, when angles are given, moves the
        outlines and the sightline marker; it computes no surface
        brightness, so render the frames first.

    Raises:
        ValueError: If a required input is missing or the image is not 2D.
    """
    ep = eyepiece()
    disk, incl, pa, dist = _resolve_disk(disk_or_image, incl_deg, pa_deg, dist_pc)
    if disk is not None:
        if wavelength_nm is None or incl is None or pa is None:
            raise ValueError(
                "rendering a disk needs wavelength_nm, incl_deg and pa_deg "
                "(a System supplies the angles)"
            )
        image = disk.surface_brightness(
            jnp.asarray(wavelength_nm, dtype=float),
            jnp.asarray(time_jd, dtype=float),
            jnp.asarray(incl, dtype=float),
            jnp.asarray(pa, dtype=float),
        )
        if pixel_scale_arcsec is None:
            pixel_scale_arcsec = getattr(disk, "pixel_scale_arcsec", None)
    elif hasattr(disk_or_image, "midplane_inc_deg") or hasattr(disk_or_image, "system"):
        raise ValueError("the system has no disk to draw")
    else:
        image = disk_or_image
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    if pixel_scale_arcsec is None:
        raise ValueError("pixel_scale_arcsec is required for a bare array")

    ny, nx = image.shape
    half_x = 0.5 * nx * float(pixel_scale_arcsec)
    half_y = 0.5 * ny * float(pixel_scale_arcsec)
    peak = float(np.nanmax(image)) if vmax is None else float(vmax)
    floor = peak / float(dynamic_range) if peak > 0.0 else 1e-30
    base = ep.imshow_log(
        image,
        ax=ax,
        extent=(-half_x, half_x, -half_y, half_y),
        floor=floor,
        vmin=floor,
        vmax=max(peak, 10.0 * floor),
        colorbar=colorbar,
        cbar_label="disk contrast per pixel",
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )
    ax = base.ax
    ax.set_xlim(half_x, -half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_xlabel(_SKY_LABELS[0])
    ax.set_ylabel(_SKY_LABELS[1])

    artists = dict(base.artists)
    star = ax.scatter(
        [0.0], [0.0], s=90, marker="*", color=_style.role("star"), zorder=5
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
                [], [], color=_style.role("disk"), lw=0.9, ls="--", alpha=0.9
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
            edgecolors=_style.neutral(0.6),
            linewidths=1.4,
            zorder=6,
        )
        marker.set_gid("sightline")
        scatters.append(marker)
    artists["scatter"] = scatters

    def place(incl_now, pa_now):
        """Move the outlines and the sightline marker to an orientation."""
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

    def update(new_image, incl_deg=None, pa_deg=None):
        """Show a new image; move the outlines when angles are given."""
        base.update(np.asarray(new_image, dtype=float))
        if incl_deg is None and pa_deg is None:
            return
        state["incl"] = state["incl"] if incl_deg is None else incl_deg
        state["pa"] = state["pa"] if pa_deg is None else pa_deg
        if state["incl"] is not None and state["pa"] is not None:
            place(float(state["incl"]), float(state["pa"]))

    return ep.PlotResult(ax=ax, artists=artists, update=update)


class _GeometryPanel:
    """The artists of a disk geometry panel, placed for one inclination."""

    def __init__(self, ax, radii, pa, grain_radius, thickness, labels, inset):
        self.ax = ax
        self.radii = radii
        self.pa = pa
        self.minor = _geometry.minor_axis_direction(pa)
        self.grain_radius = grain_radius
        self.thickness = thickness
        self.big = radii[1]
        disk_color = _style.role("disk")
        scenery = _style.neutral(0.45)
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
        (self.sightline,) = ax.plot([], [], color=_style.neutral(0.6), lw=0.9)
        self.sightline.set_gid("sightline")
        self.path = None
        if thickness is not None:
            (self.path,) = ax.plot([], [], color=disk_color, lw=4.0, alpha=0.8)
            self.path.set_gid("sightline/path")
        (self.forward,) = ax.plot([], [], color=scenery, lw=0.9, ls="--")
        self.forward.set_gid("forward")
        self.star = ax.scatter(
            [0.0], [0.0], s=170, marker="*", color=_style.role("star"), zorder=5
        )
        self.star.set_gid("star")
        self.grain = ax.scatter(
            [0.0], [0.0], s=40, color=disk_color, zorder=6, edgecolors="none"
        )
        self.grain.set_gid("grain")
        self.incident = Arrow(
            ax, (0, 0), (0, 0), color=_style.role("star"), gid="incident"
        )
        self.scattered = Arrow(ax, (0, 0), (0, 0), color=disk_color, gid="scattered")
        self.observer = Arrow(
            ax,
            (1.05 * R, 0.95 * R),
            (1.4 * R, 0.95 * R),
            color=_style.neutral(0.75),
            gid="observer",
        )
        self.theta = AngleArc(ax, color=_style.text(), gid="scattering_angle")
        self.incl_arc = AngleArc(ax, color=_style.neutral(0.75), gid="inclination")
        self.texts = [self.observer.artist, self.incident.artist]
        self.texts += [self.scattered.artist, self.theta.text, self.incl_arc.text]
        self.near_label = self.far_label = None
        if labels:
            common = {"fontsize": "small", "ha": "center", "va": "center"}
            self.near_label = ax.text(0, 0, "near side", color=disk_color, **common)
            self.near_label.set_gid("label/near_side")
            self.far_label = ax.text(0, 0, "far side", color=disk_color, **common)
            self.far_label.set_gid("label/far_side")
            observer = ax.text(
                1.22 * R,
                1.0 * R,
                "to observer",
                color=_style.neutral(0.75),
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
            for text in self.texts[-4:]:
                halo(text)
        self.inset = None
        if inset is not None:
            self.inset = GrainInset(ax, inset, grain_color=disk_color)

    def side(self, points):
        """Side-view ``(z, u)`` pairs of sky-frame points."""
        z, u = _geometry.to_side(points, self.minor)
        return np.stack([z, u], axis=-1)

    def place(self, incl):
        """Put every inclination-dependent artist where ``incl`` puts it."""
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
        self.sightline.set_data([-1.2 * R, 1.42 * R], [grain[1], grain[1]])
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
            grain + np.array([0.04 * R, 0.0]), grain + np.array([0.42 * R, 0.0])
        )
        self.forward.set_data(*np.stack([grain, grain + 0.3 * R * k_in]).T)
        self.theta.set(grain, k_in, np.array([1.0, 0.0]), 0.16 * R, r"$\Theta$", 1.9)
        self.incl_arc.set(
            (0.0, 0.0),
            np.array([0.0, 1.0]),
            d,
            0.32 * R,
            rf"$i$ = {float(incl):.0f}$^\circ$",
            1.4,
        )
        if self.near_label is not None:
            self.near_label.set_position(tuple(1.14 * r_out * d))
            self.far_label.set_position(tuple(-1.14 * r_out * d))
        if self.inset is not None:
            self.inset.set(k_in, np.array([1.0, 0.0]))

    def artists(self):
        """The panel's artists, grouped by artist-vocabulary key."""
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
    pa_deg=None,
    radii_AU=None,
    grain_radius_AU=None,
    thickness_AU=None,
    inset=True,
    inset_bounds=None,
    labels=True,
    ax=None,
):
    """Draw a disk edge-on to the reader: midplane, observer and one sightline.

    The panel is the plane holding the line of sight (horizontal, observer
    to the right along ``+z``) and the disk's projected minor axis
    (vertical); the line of nodes points out of the page. The midplane is
    the line at the inclination from the sky plane, from the inner to the
    outer truncation radius on each side, with its near half on the
    observer's side, as the library's disk kernels place it.

    One grain sits in the midplane at ``grain_radius_AU`` along the minor
    axis. The panel draws the sightline through it, the starlight arriving
    at it, the light it scatters toward the observer, and the scattering
    angle between the two, which the disk kernels evaluate their phase
    functions at. For a grain on the minor axis that angle is
    ``90 - i`` on the near side (forward scattering) and ``90 + i`` on the
    far side. An inset magnifies the grain and sets the scattering angle
    beside its supplement, the illumination angle.

    Args:
        system_or_disk: A ``System``/``Scene`` (its disk and midplane
            orientation are used) or an ``AbstractDisk``.
        incl_deg: Midplane inclination [deg]; defaults to the system's.
        pa_deg: Midplane position angle [deg]; defaults to the system's,
            else 0. It fixes which sky direction the vertical axis is.
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
        labels: Whether to add direct text labels.
        ax: Axes to draw into. None creates a new figure and axes.

    Returns:
        An ``eyepiece.PlotResult`` with ``"scatter"``, ``"lines"``,
        ``"text"`` (annotations include the arrows) and, with a thickness,
        ``"fill"``; inset artists are appended after the panel's own. Each
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
        ValueError: If no radii are available or the inclination is
            outside ``[0, 180]``.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt

    disk, incl, pa, _ = _resolve_disk(system_or_disk, incl_deg, pa_deg, 0.0)
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
            (0.62, 0.02, 0.36, 0.36) if near_grain else (0.02, 0.62, 0.36, 0.36)
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
    )

    def update(incl_deg):
        """Move the midplane, sightline, rays and arcs to ``incl_deg``."""
        _check_incl(incl_deg)
        panel.place(float(incl_deg))

    _check_incl(incl)
    panel.place(float(incl))
    R = radii[1]
    ax.set_xlim(-1.2 * R, 1.45 * R)
    ax.set_ylim(-1.2 * R, 1.2 * R)
    ax.set_aspect("equal")
    ax.set_xlabel(_SIDE_LABELS[0])
    ax.set_ylabel(_SIDE_LABELS[1])
    return ep.PlotResult(ax=ax, artists=panel.artists(), update=update)


def _check_incl(incl_deg):
    """Reject inclinations outside the documented ``[0, 180]`` range."""
    if not 0.0 <= float(incl_deg) <= 180.0:
        raise ValueError(f"incl_deg must lie in [0, 180], got {incl_deg}")

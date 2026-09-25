"""The system view: star, planets and disk extent in one physical frame.

skyscapes supplies what eyepiece cannot know: where each planet is at an
epoch (``System.positions`` and ``Planet.propagate``), where the disk's
midplane sits (``System.midplane_inc_deg`` / ``midplane_pa_deg`` through
the library's sky rotation), which way the observer is (``+z``), and that
RA offsets are drawn increasing to the left.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import au_to_arcsec

from skyscapes.viz import _geometry, _style
from skyscapes.viz._draw import Arrow, halo
from skyscapes.viz._require import eyepiece

_VIEWS = ("sky", "side")
_SKY_LABELS = ("RA offset [arcsec]", "Dec offset [arcsec]")
_SIDE_LABELS = (
    r"$z$, toward the observer [AU]",
    "offset along the projected minor axis [AU]",
)


def _unpack(system_or_scene):
    """The ``System`` inside a ``Scene``, or the ``System`` itself."""
    return getattr(system_or_scene, "system", system_or_scene)


def _default_labels(n):
    """Planet labels b, c, d, ... in the concatenated planet order."""
    return [chr(ord("b") + k) if k < 25 else f"p{k}" for k in range(n)]


def _planet_style(styles, label):
    """Color and marker for a planet: its SourceStyles entry, else the role."""
    if styles is not None:
        try:
            entry = styles[label]
        except KeyError:
            entry = None
        if entry is not None:
            return entry["color"], entry.get("marker", "o")
    return _style.role("planet"), "o"


def _star_color(styles):
    """The star's SourceStyles color when declared, else the star role."""
    if styles is not None:
        try:
            return styles["star"]["color"]
        except KeyError:
            pass
    return _style.role("star")


def _positions_xyz_AU(system, t_jd):
    """Star-centered sky-frame positions [AU], shape ``(K_total, 3, T)``."""
    per_planet = [
        np.asarray(p.propagate(system.trig_solver, t_jd, star=system.star)[0])
        for p in system.planets
    ]
    return np.concatenate(per_planet, axis=0)


def plot_system(
    system_or_scene,
    t_jd=None,
    *,
    view="sky",
    ax=None,
    styles=None,
    planet_labels=None,
    track_t_jd=None,
    disk_radii_AU=None,
    labels=True,
):
    """Draw a planetary system: star, planets and disk extent, in one frame.

    ``view="sky"`` is the sky-projected scene in arcsec, RA offset
    increasing to the left: planet positions at ``t_jd`` from
    ``System.positions``, optional orbit tracks over ``track_t_jd``, and the
    disk's inner and outer truncation rings projected at the system's
    midplane inclination and position angle, with the line of nodes. The
    observer looks along ``-z``, into the page.

    ``view="side"`` turns the scene edge-on to the reader, in AU: the
    horizontal axis is the line of sight with the observer to the right
    (``+z``), the vertical axis the sky-plane offset along the disk's
    projected minor axis. The disk midplane appears as a line at the
    inclination from the sky plane, the near half on the observer's side,
    and each planet sits at its true distance toward or away from the
    observer.

    The disk's orientation comes only from ``System.midplane_inc_deg``
    (in ``[0, 180]``) and ``midplane_pa_deg``; the planets' orbits carry
    their own elements and are not assumed coplanar with it. The position
    angle is the library's, not the astronomical one: it turns the line of
    nodes from ``+x`` (RA offset) toward ``+y`` (Dec offset), so with ``+x``
    east the major axis lies at astronomical PA (north through east)
    ``90 - midplane_pa_deg`` (mod 180). Within about 6 degrees of face-on
    the near-side label is left off.

    Args:
        system_or_scene: A ``skyscapes.System`` or ``skyscapes.Scene``.
        t_jd: Epoch of the planet markers [JD], scalar. Required when the
            system has planets.
        view: ``"sky"`` or ``"side"``.
        ax: Axes to draw into. None creates a new figure and axes.
        styles: Optional ``eyepiece.SourceStyles`` declaring ``"star"``
            and the planet labels, so this view matches every other figure
            of the document. Names it does not declare take the brand
            roles (star yellow, planet cyan).
        planet_labels: One label per planet, in ``System.positions`` order.
            None labels them ``b``, ``c``, ... .
        track_t_jd: Optional times [JD], shape ``(T,)``, over which to draw
            each planet's track.
        disk_radii_AU: ``(r_in, r_out)`` for the disk rings. None reads the
            truncation radii of a parametric disk and draws no rings for a
            disk that has none (a pre-rendered ``ExovistaDisk``).
        labels: Whether to add direct text labels.

    Returns:
        An ``eyepiece.PlotResult`` with ``"scatter"`` (the star, then one
        collection per planet), ``"lines"`` (disk rings and line of nodes,
        or the midplane halves side on, then tracks, and in the side view
        the sky plane), and ``"text"`` (labels
        and, in the side view, the observer arrow). Every artist carries a
        ``gid``: ``star``, ``planet/<label>``, ``track/<label>``,
        ``disk/inner``, ``disk/outer`` and ``disk/nodes`` (sky view),
        ``disk/near`` and ``disk/far`` (side view), ``sky_plane``,
        ``observer``, ``label/...``. ``update`` is None: an orbit animation
        belongs to ``orbix.viz.animate_orbit``.

    Raises:
        ValueError: If ``view`` is unknown, the labels do not match the
            planet count, or the midplane inclination is outside
            ``[0, 180]``.
        TypeError: If the system has planets and ``t_jd`` is None.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt

    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}, got {view!r}")
    system = _unpack(system_or_scene)
    n_planets = system.n_planets
    if n_planets and t_jd is None:
        raise TypeError("t_jd is required to place the planets")
    names = list(planet_labels) if planet_labels is not None else None
    if names is None:
        names = _default_labels(n_planets)
    if len(names) != n_planets:
        raise ValueError(f"{len(names)} planet labels for {n_planets} planets")

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    # An axes that already holds data (a disk image drawn first, say) keeps
    # it in view: the limits below grow to cover the old ones.
    previous = 0.0
    if ax.has_data():
        previous = float(np.max(np.abs([*ax.get_xlim(), *ax.get_ylim()])))
    radii = disk_radii_AU or _geometry.disk_radii_AU(system.disk)
    incl = float(system.midplane_inc_deg)
    if not 0.0 <= incl <= 180.0:
        raise ValueError(f"midplane_inc_deg must lie in [0, 180], got {incl}")
    pa = float(system.midplane_pa_deg)
    dist_pc = float(system.star.dist_pc)

    scatters, lines, texts = [], [], []
    extent = [0.0]
    scenery = _style.neutral(0.5)
    disk_color = _style.role("disk")

    if view == "sky":
        to_2d = _sky_projector(dist_pc)
    else:
        minor = _geometry.minor_axis_direction(pa)
        to_2d = _side_projector(minor)

    # The disk, from the library's midplane rotation: projected truncation
    # rings and the line of nodes on the sky, the two midplane halves side on.
    if radii is not None and system.disk is not None:
        nodes, near, _ = _geometry.disk_axes_sky(incl, pa)
        if view == "sky":
            for name, radius in (("inner", radii[0]), ("outer", radii[1])):
                h, v = to_2d(_geometry.ring_sky(radius, incl, pa))
                (line,) = ax.plot(h, v, color=disk_color, lw=1.3)
                line.set_gid(f"disk/{name}")
                lines.append(line)
                extent.append(float(np.max(np.abs(np.concatenate([h, v])))))
            h, v = to_2d(np.stack([-radii[1] * nodes, radii[1] * nodes]))
            (line,) = ax.plot(h, v, color=scenery, lw=0.8, ls="--")
            line.set_gid("disk/nodes")
            lines.append(line)
        else:
            for name, sign in (("near", 1.0), ("far", -1.0)):
                ends = sign * np.stack([radii[0] * near, radii[1] * near])
                h, v = to_2d(ends)
                (line,) = ax.plot(h, v, color=disk_color, lw=2.2)
                line.set_gid(f"disk/{name}")
                lines.append(line)
                extent.append(float(np.max(np.abs(np.concatenate([h, v])))))
        # Face-on, no half is nearer the observer: no near-side label.
        tilted = abs(np.sin(np.radians(incl))) >= 0.1
        if labels and tilted:
            h, v = to_2d((1.1 * radii[1] * near)[None, :])
            text = ax.text(
                float(h[0]),
                float(v[0]),
                "near side",
                color=disk_color,
                ha="center",
                va="center",
                fontsize="small",
            )
            text.set_gid("label/near_side")
            texts.append(halo(text))

    # Planets and their tracks.
    if n_planets:
        t_now = jnp.atleast_1d(jnp.asarray(t_jd, dtype=float))
        now = _positions_xyz_AU(system, t_now)[:, :, 0]
        track = None
        if track_t_jd is not None:
            track = _positions_xyz_AU(system, jnp.asarray(track_t_jd, dtype=float))
        for k, name in enumerate(names):
            color, marker = _planet_style(styles, name)
            if track is not None:
                h, v = to_2d(np.moveaxis(track[k], 0, -1))
                (line,) = ax.plot(h, v, color=color, lw=0.9, alpha=0.55)
                line.set_gid(f"track/{name}")
                lines.append(line)
                extent.append(float(np.max(np.abs(np.concatenate([h, v])))))
            h, v = to_2d(now[k][None, :])
            dot = ax.scatter(h, v, s=42, color=color, marker=marker, zorder=4)
            dot.set_gid(f"planet/{name}")
            scatters.append(dot)
            extent.append(float(np.max(np.abs(np.concatenate([h, v])))))
            if labels:
                text = ax.annotate(
                    name,
                    (float(h[0]), float(v[0])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=color,
                    fontsize="small",
                )
                text.set_gid(f"label/planet/{name}")
                texts.append(halo(text))

    star = ax.scatter(
        [0.0], [0.0], s=150, marker="*", color=_star_color(styles), zorder=5
    )
    star.set_gid("star")
    scatters.insert(0, star)

    half = 1.15 * max(extent) if max(extent) > 0.0 else 1.0
    half = max(half, previous)
    if view == "sky":
        ax.set_xlim(half, -half)
        ax.set_ylim(-half, half)
        ax.set_xlabel(_SKY_LABELS[0])
        ax.set_ylabel(_SKY_LABELS[1])
    else:
        texts += _side_furniture(ax, half, lines, labels)
        ax.set_xlim(-half, 1.3 * half)
        ax.set_ylim(-half, half)
        ax.set_xlabel(_SIDE_LABELS[0])
        ax.set_ylabel(_SIDE_LABELS[1])
    ax.set_aspect("equal")

    artists = {"scatter": scatters, "lines": lines}
    if texts:
        artists["text"] = texts
    return ep.PlotResult(ax=ax, artists=artists)


def _sky_projector(dist_pc):
    """Sky-frame AU points to (RA offset, Dec offset) arcsec."""

    def project(points):
        p = np.asarray(points, dtype=float)
        return (
            np.asarray(au_to_arcsec(p[..., 0], dist_pc)),
            np.asarray(au_to_arcsec(p[..., 1], dist_pc)),
        )

    return project


def _side_projector(minor):
    """Sky-frame AU points to (z, minor-axis offset) AU."""

    def project(points):
        return _geometry.to_side(points, minor)

    return project


def _side_furniture(ax, half, lines, labels):
    """The sky plane and the observer arrow of a side view."""
    scenery = _style.neutral(0.45)
    (plane,) = ax.plot([0.0, 0.0], [-half, half], color=scenery, lw=0.8, ls=":")
    plane.set_gid("sky_plane")
    lines.append(plane)
    arrow = Arrow(
        ax,
        (0.95 * half, 0.85 * half),
        (1.25 * half, 0.85 * half),
        color=_style.neutral(0.75),
        gid="observer",
    )
    texts = [arrow.artist]
    if labels:
        text = ax.text(
            1.1 * half,
            0.93 * half,
            "to observer",
            color=_style.neutral(0.75),
            ha="center",
            va="bottom",
            fontsize="small",
        )
        text.set_gid("label/observer")
        halo(text)
        sky = ax.text(
            0.02 * half,
            -0.95 * half,
            "sky plane",
            color=scenery,
            ha="left",
            va="bottom",
            fontsize="small",
        )
        sky.set_gid("label/sky_plane")
        halo(sky)
        texts += [text, sky]
    return texts

"""One grain in its scattering plane: the scattering and illumination angles.

This is the drawing the geometry views magnify in their grain insets, at
full panel size and with the star and the observer at the ray ends. A
figure that teaches the two angles before it places the grain in a cloud or
a disk can therefore show the same construction the later view carries.
"""

from __future__ import annotations

import numpy as np

from skyscapes.viz import _style
from skyscapes.viz._draw import (
    ALPHA_RADIUS,
    THETA_RADIUS,
    ScatteringAngle,
    scattering_plane,
)
from skyscapes.viz._require import eyepiece

# The drawing's rays are about one unit long; the glyphs and labels sit
# beyond their ends.
_HALF_SPAN = 1.6


def _plane(k_in, k_out):
    """2D in-plane directions for 2D or 3D ``k_in`` and ``k_out``."""
    k_in = np.asarray(k_in, dtype=float)
    k_out = np.asarray(k_out, dtype=float)
    if k_in.shape != k_out.shape or k_in.shape not in ((2,), (3,)):
        raise ValueError(
            "k_in and k_out must both have shape (2,) or both (3,), got "
            f"{k_in.shape} and {k_out.shape}"
        )
    if not (np.linalg.norm(k_in) > 0.0 and np.linalg.norm(k_out) > 0.0):
        raise ValueError("k_in and k_out must be nonzero directions")
    if k_in.shape == (2,):
        return k_in, k_out
    out_ref = k_out[:2] if np.linalg.norm(k_out[:2]) > 0.0 else np.array([1.0, 0.0])
    return scattering_plane(k_in, k_out, out_ref, k_in[:2])


def plot_scattering_angle(
    k_in,
    k_out,
    *,
    grain_color=None,
    star=True,
    observer=True,
    labels=True,
    values="corner",
    theta_radius=THETA_RADIUS,
    alpha_radius=ALPHA_RADIUS,
    gid_prefix="",
    ax=None,
):
    """Draw one grain in its scattering plane, with both angles marked.

    Starlight arrives at the grain along ``k_in`` (the propagation
    direction, star to grain) and leaves it toward the observer along
    ``k_out``. The scattering angle ``Theta`` is measured from the forward
    continuation of ``k_in`` (dashed) to ``k_out``, so ``Theta = 0`` is
    forward scattering; the illumination angle ``alpha = 180 - Theta`` is
    measured from the direction back toward the star to ``k_out``. For a
    disk seen from far along ``+z``, ``cos(Theta) = k_in . k_out`` is the
    angle the disk kernels evaluate their phase functions at.

    Two-element directions are drawn in the page plane as given. Three-
    element directions are laid into the plane they span, turned so
    ``k_out`` points along its own ``(x, y)`` projection (``+x`` when that
    is zero) and ``k_in`` falls on the side its projection takes. That is
    how the top view of ``plot_local_zodi_geometry`` turns its grain inset,
    so the same 3D pair drawn here and in that inset has the same
    orientation.

    The panel carries no axes: the drawing is directions only, in units of
    the ray length, with the grain at the origin.

    Args:
        k_in: Incident propagation direction (star to grain), shape ``(2,)``
            or ``(3,)``; need not be unit.
        k_out: Scattered direction (grain to observer), the same shape.
        grain_color: Color of the grain and its scattered ray. None takes
            the disk role color.
        star: Whether to mark the star at the tail of the incident ray.
        observer: Whether to mark the observer at the head of the scattered
            ray.
        labels: Whether to label the incident ray, the scattered ray and
            the forward continuation.
        values: Where to print the two angle values: ``"corner"`` (the
            upper and lower left corners), ``"arcs"`` (on the arc labels,
            which then read ``Theta = 62`` and ``alpha = 118`` degrees and
            grow away from the grain), or None (the arcs keep their bare
            symbols and nothing else is printed). The corner texts keep
            their gids in every case and are empty when unused.
        theta_radius: Radius of the scattering-angle arc, in units of the
            ray length. Its label sits a fixed gap beyond it.
        alpha_radius: Radius of the illumination-angle arc, as
            ``theta_radius``. A narrow illumination wedge reads better on
            a larger arc.
        gid_prefix: Prefix for every gid, joined with ``/``, so
            ``"inset"`` gives the gids the geometry views' insets carry
            (``inset/incident`` and so on). The empty default adds none.
        ax: Axes to draw into. None creates a new figure and axes.

    Returns:
        An ``eyepiece.PlotResult`` with ``"text"`` (the two ray arrows are
        annotations; the arc labels; the angle values; the ray labels),
        ``"lines"`` (``forward`` and the two arcs) and ``"scatter"``
        (``grain``, ``star``, ``observer``). Gids: ``incident``,
        ``scattered``, ``forward``, ``grain``, ``scattering_angle``,
        ``illumination_angle`` (each arc's text adds ``/label``, its value
        ``/value``), ``star``, ``observer`` and ``label/incident``,
        ``label/scattered``, ``label/forward``, each under ``gid_prefix``
        when one is given. The grain insets of the geometry views carry the
        same gids under ``inset/``.
        ``update(k_in, k_out)`` redraws for a new pair of directions.

    Raises:
        ValueError: If the directions are zero or not both 2D or both 3D,
            ``values`` is unknown, or an arc radius is not positive.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt

    k_in_2d, k_out_2d = _plane(k_in, k_out)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    color = _style.role("disk") if grain_color is None else grain_color
    drawing = ScatteringAngle(
        ax,
        grain_color=color,
        value_size="small",
        star=star,
        observer=observer,
        labels=labels,
        values=values,
        theta_radius=theta_radius,
        alpha_radius=alpha_radius,
        gid_prefix=gid_prefix,
    )
    drawing.set(k_in_2d, k_out_2d)
    ax.set_xlim(-_HALF_SPAN, _HALF_SPAN)
    ax.set_ylim(-_HALF_SPAN, _HALF_SPAN)
    ax.set_aspect("equal")
    ax.set_axis_off()

    def update(k_in, k_out):
        """Redraw the grain for a new pair of directions."""
        drawing.set(*_plane(k_in, k_out))

    return ep.PlotResult(ax=ax, artists=drawing.artists(), update=update)

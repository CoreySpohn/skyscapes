"""Small drawing devices the views share: arrows, angle arcs, a grain inset.

Each device creates its artists once and moves them through ``set`` so a
view's ``update`` mutates existing artists and never adds new ones. Every
artist carries a ``gid``, which is how a caller finds one part of a view.
"""

from __future__ import annotations

import numpy as np

from skyscapes.viz import _style


def halo(text):
    """Outline ``text`` so it reads over whatever lies under it.

    On an axes that holds an image the outline is black or white, whichever
    contrasts more with the text color, because the image, not the axes
    background, is what the label sits on. Otherwise the outline takes the
    facecolor of the axes the text belongs to.
    """
    from matplotlib import patheffects
    from matplotlib.colors import to_rgb

    ax = text.axes
    if ax is not None and ax.images:
        r, g, b = to_rgb(text.get_color())
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        foreground = "black" if luminance > 0.45 else "white"
    elif ax is not None:
        foreground = ax.get_facecolor()
    else:
        import matplotlib as mpl

        foreground = mpl.rcParams["axes.facecolor"]
    text.set_path_effects(
        [patheffects.withStroke(linewidth=2.5, foreground=foreground)]
    )
    return text


class Arrow:
    """A straight arrow from ``start`` to ``end`` (an ``Annotation``)."""

    def __init__(self, ax, start, end, *, color, gid, lw=1.4, label=None):
        """Draw the arrow, optionally with a text label at its head."""
        self.artist = ax.annotate(
            "",
            xy=tuple(end),
            xytext=tuple(start),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": lw,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "mutation_scale": 11,
            },
            annotation_clip=False,
        )
        self.artist.set_gid(gid)

    def set(self, start, end):
        """Move the arrow."""
        self.artist.xy = tuple(end)
        self.artist.set_position(tuple(start))


def _wrap(angle):
    """Wrap an angle in radians to ``(-pi, pi]``."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class AngleArc:
    """The shorter arc between two directions, with a label at its middle."""

    def __init__(self, ax, *, color, gid, lw=1.1, fontsize=None):
        """Create the (empty) arc and label; place them with ``set``."""
        (self.line,) = ax.plot([], [], color=color, lw=lw, solid_capstyle="round")
        self.line.set_gid(gid)
        self.text = ax.text(
            0.0,
            0.0,
            "",
            color=color,
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        self.text.set_gid(gid + "/label")
        halo(self.text)

    def set(
        self, center, v_from, v_to, radius, label, label_scale=1.45, label_dir=None
    ):
        """Draw the arc about ``center`` from direction ``v_from`` to ``v_to``.

        The label sits at ``label_scale * radius`` along the arc's middle
        direction, or along ``label_dir`` when the middle is crowded.
        """
        a0 = np.arctan2(v_from[1], v_from[0])
        sweep = _wrap(np.arctan2(v_to[1], v_to[0]) - a0)
        t = a0 + np.linspace(0.0, 1.0, 48) * sweep
        cx, cy = center
        self.line.set_data(cx + radius * np.cos(t), cy + radius * np.sin(t))
        mid = a0 + 0.5 * sweep
        if label_dir is not None:
            mid = np.arctan2(label_dir[1], label_dir[0])
        self.text.set_position(
            (
                cx + label_scale * radius * np.cos(mid),
                cy + label_scale * radius * np.sin(mid),
            )
        )
        self.text.set_text(label)

    def clear(self):
        """Empty the arc and its label, for a configuration with no angle."""
        self.line.set_data([], [])
        self.text.set_text("")


def unit(v):
    """``v`` scaled to unit length."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class GrainInset:
    """A magnified grain in its scattering plane: the two angles defined.

    The incident light arrives along ``k_in`` (star to grain), and the
    scattered light leaves toward the observer along ``k_out``. The
    scattering angle is measured from the forward continuation of ``k_in``
    to ``k_out``; the illumination angle, its supplement, from the
    direction back toward the star to ``k_out``. For a disk seen from far
    along ``+z`` this is the angle the disk kernels evaluate their phase
    functions at: their ``cos_phi`` is the line-of-sight coordinate over the
    star-grain distance, which is ``k_in . k_out``.
    """

    def __init__(self, ax, bounds, *, grain_color, gid_prefix="inset"):
        """Create the inset axes and its artists; place them with ``set``."""
        self.ax = ax.inset_axes(bounds)
        self.ax.set_gid(gid_prefix)
        inset = self.ax
        inset.set_xlim(-1.35, 1.35)
        inset.set_ylim(-1.35, 1.35)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(_style.neutral(0.35))
            spine.set_linewidth(0.8)
        scenery = _style.neutral(0.55)
        self.incident = Arrow(
            inset,
            (-1.0, 0.0),
            (0.0, 0.0),
            color=_style.role("star"),
            gid=f"{gid_prefix}/incident",
        )
        self.outgoing = Arrow(
            inset,
            (0.0, 0.0),
            (1.0, 0.0),
            color=grain_color,
            gid=f"{gid_prefix}/scattered",
        )
        (self.forward,) = inset.plot([], [], color=scenery, lw=0.9, ls="--")
        self.forward.set_gid(f"{gid_prefix}/forward")
        self.theta = AngleArc(
            inset, color=_style.text(), gid=f"{gid_prefix}/scattering_angle"
        )
        self.alpha = AngleArc(
            inset, color=scenery, gid=f"{gid_prefix}/illumination_angle"
        )
        self.grain = inset.scatter(
            [0.0], [0.0], s=46, color=grain_color, zorder=5, edgecolors="none"
        )
        self.grain.set_gid(f"{gid_prefix}/grain")
        small = {"fontsize": "x-small", "transform": inset.transAxes, "ha": "left"}
        self.theta_value = inset.text(
            0.04, 0.96, "", color=_style.text(), va="top", **small
        )
        self.theta_value.set_gid(f"{gid_prefix}/scattering_angle/value")
        self.alpha_value = inset.text(
            0.04, 0.04, "", color=scenery, va="bottom", **small
        )
        self.alpha_value.set_gid(f"{gid_prefix}/illumination_angle/value")

    def set(self, k_in, k_out):
        """Draw the grain for incident direction ``k_in`` and outgoing ``k_out``.

        Both are 2D directions in the inset's plane; they need not be unit.
        """
        k_in = unit(k_in)
        k_out = unit(k_out)
        self.incident.set(-1.05 * k_in, -0.08 * k_in)
        self.outgoing.set(0.08 * k_out, 1.05 * k_out)
        self.forward.set_data([0.0, 0.8 * k_in[0]], [0.0, 0.8 * k_in[1]])
        theta = np.degrees(np.arccos(np.clip(k_in @ k_out, -1.0, 1.0)))
        self.theta.set((0.0, 0.0), k_in, k_out, 0.5, r"$\Theta$", 1.45)
        self.alpha.set((0.0, 0.0), -k_in, k_out, 0.3, r"$\alpha$", 1.9)
        self.theta_value.set_text(rf"scattering $\Theta$ = {theta:.0f}$^\circ$")
        self.alpha_value.set_text(
            rf"illumination $\alpha$ = {180.0 - theta:.0f}$^\circ$"
        )

    def artists(self):
        """Every artist the inset owns, grouped by artist-vocabulary key."""
        return {
            "text": [
                self.incident.artist,
                self.outgoing.artist,
                self.theta.text,
                self.alpha.text,
                self.theta_value,
                self.alpha_value,
            ],
            "lines": [self.forward, self.theta.line, self.alpha.line],
            "scatter": [self.grain],
        }


def scattering_plane(k_in, k_out, reference_out_2d, reference_in_2d=None):
    """Coordinates of two 3D directions in the plane they span.

    The plane is oriented so ``k_out`` points along ``reference_out_2d``
    (typically its projection in the main panel), and, when
    ``reference_in_2d`` is given, so ``k_in`` falls on the same side of it
    as that projection does, keeping the inset turned the way the reader's
    eye already is.

    Args:
        k_in: Incident direction, shape ``(3,)``.
        k_out: Outgoing direction, shape ``(3,)``.
        reference_out_2d: The 2D direction ``k_out`` should point along.
        reference_in_2d: Optional 2D direction whose side of
            ``reference_out_2d`` ``k_in`` should take.

    Returns:
        ``(k_in_2d, k_out_2d)``, each shape ``(2,)``; the angle between
        them equals the 3D angle.
    """
    k_in = unit(k_in)
    k_out = unit(k_out)
    along = k_in @ k_out
    perp = k_in - along * k_out
    across = float(np.linalg.norm(perp))
    ref = unit(reference_out_2d)
    ref_perp = np.array([-ref[1], ref[0]])
    if reference_in_2d is not None and np.dot(reference_in_2d, ref_perp) < 0.0:
        ref_perp = -ref_perp
    return along * ref + across * ref_perp, ref

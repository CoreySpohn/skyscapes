"""The local zodiacal-light geometry: an observer inside the Solar-system cloud.

``LeinertZodi`` reads its brightness off two angles of the look direction,
its ecliptic latitude and its helio-ecliptic longitude difference from the
Sun. This view draws what those two inputs mean: the Sun, the observer
inside the cloud, the half-ray the brightness integrates along, and, for
one grain on that ray, the scattering angle between the sunlight it
receives and the light it sends to the observer. Computing the two angles
for a real target and epoch is the observatory model's job
(``orbix.observatory``); this view takes them as given.
"""

from __future__ import annotations

import numpy as np

from skyscapes.viz import _geometry, _style
from skyscapes.viz._draw import AngleArc, Arrow, GrainInset, halo, scattering_plane
from skyscapes.viz._require import eyepiece

_VIEWS = ("top", "side")


def _check(ecliptic_lat_deg, solar_lon_deg):
    """Reject angles outside the domain the Leinert inputs are defined on."""
    if not -90.0 <= float(ecliptic_lat_deg) <= 90.0:
        raise ValueError(
            f"ecliptic_lat_deg must lie in [-90, 90], got {ecliptic_lat_deg}"
        )
    if not 0.0 <= float(solar_lon_deg) <= 180.0:
        raise ValueError(f"solar_lon_deg must lie in [0, 180], got {solar_lon_deg}")


class _ZodiPanel:
    """The artists of a local-zodi geometry panel, placed for one look."""

    def __init__(self, ax, view, R, ray_length, grain_distance, labels, inset):
        self.ax = ax
        self.view = view
        self.R = R
        self.ray_length = ray_length
        self.grain_distance = grain_distance
        zodi = _style.local_zodi()
        star = _style.role("star")
        scenery = _style.neutral(0.45)
        observer_color = _style.neutral(0.8)

        self.lines, self.texts, self.scatters = [], [], []
        if view == "top":
            t = np.linspace(0.0, 2.0 * np.pi, 361)
            (orbit,) = ax.plot(R * np.cos(t), R * np.sin(t), color=scenery)
            orbit.set(lw=0.8, ls=":", gid="observer_orbit")
            self.lines.append(orbit)
            self.sun_xy = np.zeros(2)
            self.observer_xy = np.array([R, 0.0])
        else:
            (ecliptic,) = ax.plot([], [], color=scenery, lw=0.8, ls=":")
            ecliptic.set_gid("ecliptic")
            self.lines.append(ecliptic)
            self.ecliptic = ecliptic
            self.observer_xy = np.zeros(2)
        (self.to_sun,) = ax.plot([], [], color=scenery, lw=0.8, ls="--")
        self.to_sun.set_gid("to_sun")
        (self.ray,) = ax.plot([], [], color=zodi, lw=1.3)
        self.ray.set_gid("sightline")
        self.lines += [self.to_sun, self.ray]

        self.sun = ax.scatter([0.0], [0.0], s=190, marker="*", color=star, zorder=5)
        self.sun.set_gid("sun")
        if view == "side":
            self.sun.set_facecolors("none")
            self.sun.set_edgecolors(star)
        self.observer = ax.scatter(
            [self.observer_xy[0]],
            [self.observer_xy[1]],
            s=45,
            marker="s",
            color=observer_color,
            zorder=6,
        )
        self.observer.set_gid("observer")
        self.grain = ax.scatter(
            [0.0], [0.0], s=40, color=zodi, zorder=6, edgecolors="none"
        )
        self.grain.set_gid("grain")
        self.scatters += [self.sun, self.observer, self.grain]

        self.incident = None
        self.scattered = None
        if view == "top":
            self.incident = Arrow(ax, (0, 0), (0, 0), color=star, gid="incident")
            self.scattered = Arrow(ax, (0, 0), (0, 0), color=zodi, gid="scattered")
        self.angle = AngleArc(ax, color=_style.neutral(0.8), gid="look_angle")
        self.texts.append(self.angle.text)
        self.lines.append(self.angle.line)
        if self.incident is not None:
            self.texts += [self.incident.artist, self.scattered.artist]
        self.ray_label = None
        if labels:
            self.ray_label = halo(
                ax.text(0, 0, "sightline", color=zodi, fontsize="small", ha="left")
            )
            self.ray_label.set_gid("label/sightline")
            self.texts.append(self.ray_label)

        self.readout = ax.text(
            0.98,
            0.02,
            "",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=_style.neutral(0.8),
            fontsize="small",
        )
        self.readout.set_gid("label/angles")
        self.texts.append(self.readout)
        self.sun_label = None
        if labels:
            observer_label = ax.annotate(
                "observer",
                tuple(self.observer_xy),
                xytext=(6, -12),
                textcoords="offset points",
                color=observer_color,
                fontsize="small",
            )
            observer_label.set_gid("label/observer")
            halo(observer_label)
            self.sun_label = ax.annotate(
                "Sun" if view == "top" else "Sun, projected",
                (0.0, 0.0),
                xytext=(6, 6),
                textcoords="offset points",
                color=star,
                fontsize="small",
            )
            self.sun_label.set_gid("label/sun")
            halo(self.sun_label)
            self.texts += [observer_label, self.sun_label]
        self.inset = None
        if inset is not None:
            self.inset = GrainInset(ax, inset, grain_color=zodi)

    def place(self, ecliptic_lat_deg, solar_lon_deg):
        """Put every look-dependent artist where the two angles put it."""
        R = self.R
        look = _geometry.zodi_sightline(ecliptic_lat_deg, solar_lon_deg)
        observer3 = np.array([R, 0.0, 0.0])
        grain3 = observer3 + self.grain_distance * look
        sun_dir = np.array([-1.0, 0.0, 0.0])
        elongation = float(np.degrees(np.arccos(np.clip(look @ sun_dir, -1.0, 1.0))))

        if self.view == "top":
            proj = look[:2]
            obs = self.observer_xy
            grain = grain3[:2]
            end = obs + self.ray_length * proj
            self.ray.set_data([obs[0], end[0]], [obs[1], end[1]])
            self.to_sun.set_data([obs[0], 0.0], [obs[1], 0.0])
            self.grain.set_offsets([grain])
            k_in = grain / np.linalg.norm(grain)
            self.incident.set(0.1 * R * k_in, grain - 0.04 * R * k_in)
            back = obs - grain
            back_len = np.linalg.norm(back)
            if back_len > 0.0:
                self.scattered.set(
                    grain, grain + 0.45 * min(back_len, R) * back / back_len
                )
            if np.linalg.norm(proj) > 1e-9:
                self.angle.set(
                    obs,
                    sun_dir[:2],
                    proj,
                    0.3 * R,
                    rf"$\Delta\lambda_\odot$ = {float(solar_lon_deg):.0f}$^\circ$",
                    1.8,
                )
            out_2d = back if np.linalg.norm(back[:2]) > 0.0 else np.array([1.0, 0.0])
            in_2d = k_in
        else:
            beta = np.radians(ecliptic_lat_deg)
            dlon = np.radians(solar_lon_deg)
            direction = np.array([np.cos(beta), np.sin(beta)])
            end = self.ray_length * direction
            self.ray.set_data([0.0, end[0]], [0.0, end[1]])
            sun_along = R * np.cos(dlon)
            self.sun.set_offsets([[sun_along, 0.0]])
            if self.sun_label is not None:
                self.sun_label.xy = (sun_along, 0.0)
            self.to_sun.set_data([0.0, sun_along], [0.0, 0.0])
            reach = max(abs(sun_along), self.ray_length) * 1.1
            self.ecliptic.set_data([-reach, reach], [0.0, 0.0])
            grain = self.grain_distance * direction
            self.grain.set_offsets([grain])
            self.angle.set(
                (0.0, 0.0),
                np.array([1.0, 0.0]),
                direction,
                0.3 * R,
                rf"$\beta$ = {float(ecliptic_lat_deg):.0f}$^\circ$",
                1.9,
            )
            out_2d = -direction
            in_2d = direction
        if self.ray_label is not None:
            self.ray_label.set_position((end[0], end[1]))
        self.readout.set_text(
            rf"$\beta$ = {float(ecliptic_lat_deg):.0f}$^\circ$, "
            rf"$\Delta\lambda_\odot$ = {float(solar_lon_deg):.0f}$^\circ$, "
            f"solar elongation = {elongation:.0f}$^\\circ$"
        )
        if self.inset is not None:
            k_in3 = grain3 / np.linalg.norm(grain3)
            k_in_2d, k_out_2d = scattering_plane(k_in3, -look, out_2d, in_2d)
            self.inset.set(k_in_2d, k_out_2d)

    def artists(self):
        """The panel's artists, grouped by artist-vocabulary key."""
        out = {
            "scatter": list(self.scatters),
            "lines": list(self.lines),
            "text": list(self.texts),
        }
        if self.inset is not None:
            for key, values in self.inset.artists().items():
                out[key] = out[key] + values
        return out


def plot_local_zodi_geometry(
    ecliptic_lat_deg,
    solar_lon_deg,
    *,
    view="top",
    observer_distance_AU=1.0,
    grain_distance_AU=None,
    ray_length_AU=None,
    inset=True,
    inset_bounds=None,
    labels=True,
    ax=None,
):
    """Draw the geometry behind a local zodiacal-light lookup.

    The two inputs are those of ``LeinertZodi.spec_flux_density``: the
    look direction's ecliptic latitude ``beta`` and its helio-ecliptic
    longitude difference from the Sun ``Delta lambda`` (0 toward the Sun,
    180 away from it). The observer sits inside the cloud at
    ``observer_distance_AU`` from the Sun; the sightline is the half-ray
    from the observer along the look direction, which is all the
    brightness integrates over.

    ``view="top"`` looks down on the ecliptic from its north pole, in AU:
    the Sun at the origin, the observer on ``+x``, the sightline's
    projection at ``Delta lambda`` from the Sun direction, and one grain on
    the sightline with the sunlight it receives and the light it scatters
    back to the observer. The scattering angle between the two is drawn
    only in the inset, in the grain's own scattering plane, because the
    top view foreshortens it whenever ``beta`` is not zero.
    ``view="side"`` stands in the vertical plane of the sightline, the
    observer at the origin and the ecliptic horizontal, so the sightline
    rises at ``beta``; the Sun is shown projected onto that plane. Both
    views print ``beta``, ``Delta lambda`` and the 3D solar elongation,
    which differ off the ecliptic.

    The sign of ``Delta lambda`` (east or west of the Sun) is drawn toward
    ``+y``; the table is symmetric in it. No density is drawn: the cloud
    is everywhere around the observer, and ``LeinertZodi`` is an empirical
    table of the integrated brightness, not a density model.

    Args:
        ecliptic_lat_deg: Ecliptic latitude of the look direction [deg],
            in ``[-90, 90]``.
        solar_lon_deg: Helio-ecliptic longitude difference [deg], in
            ``[0, 180]``.
        view: ``"top"`` or ``"side"``.
        observer_distance_AU: Observer's distance from the Sun [AU].
        grain_distance_AU: Distance of the marked grain along the
            sightline [AU]. None takes half the observer distance.
        ray_length_AU: Drawn length of the sightline [AU]. None takes
            1.6 times the observer distance.
        inset: Whether to draw the grain inset, in the grain's scattering
            plane.
        inset_bounds: The inset's ``[x0, y0, width, height]`` in axes
            fractions. None puts it in a lower-left corner the sightline
            never enters (upper left in a side view that looks below the
            ecliptic).
        labels: Whether to add direct text labels.
        ax: Axes to draw into. None creates a new figure and axes.

    Returns:
        An ``eyepiece.PlotResult`` with ``"scatter"`` (``sun``,
        ``observer``, ``grain``), ``"lines"`` and ``"text"`` (arrows are
        annotations), inset artists appended. Every artist carries a
        ``gid``. ``update(ecliptic_lat_deg, solar_lon_deg)`` moves the
        sightline, grain, rays, arcs and readout, leaving the observer,
        the observer's orbit and the fixed labels untouched, for a sweep
        over the year.

    Raises:
        ValueError: If ``view`` is unknown or an angle is out of range.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt

    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}, got {view!r}")
    _check(ecliptic_lat_deg, solar_lon_deg)
    R = float(observer_distance_AU)
    grain_distance = 0.5 * R if grain_distance_AU is None else grain_distance_AU
    ray_length = 1.6 * R if ray_length_AU is None else ray_length_AU

    if inset_bounds is None:
        below = view == "side" and float(ecliptic_lat_deg) < 0.0
        inset_bounds = (0.02, 0.62, 0.34, 0.34) if below else (0.02, 0.1, 0.34, 0.34)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    panel = _ZodiPanel(
        ax,
        view,
        R,
        float(ray_length),
        float(grain_distance),
        labels,
        inset_bounds if inset else None,
    )
    panel.place(float(ecliptic_lat_deg), float(solar_lon_deg))

    reach = R + ray_length
    if view == "top":
        ax.set_xlim(-1.25 * R, 1.05 * reach)
        ax.set_ylim(-0.75 * reach, 0.75 * reach)
        ax.set_xlabel(r"$x$, Sun to observer [AU]")
        ax.set_ylabel(r"$y$ [AU]")
    else:
        ax.set_xlim(-1.15 * max(R, ray_length), 1.15 * max(R, ray_length))
        ax.set_ylim(-0.75 * max(R, ray_length), 1.05 * ray_length)
        ax.set_xlabel("distance along the sightline's ecliptic projection [AU]")
        ax.set_ylabel("height above the ecliptic [AU]")
    ax.set_aspect("equal")

    def update(ecliptic_lat_deg, solar_lon_deg):
        """Move the look-dependent artists to a new pair of angles."""
        _check(ecliptic_lat_deg, solar_lon_deg)
        panel.place(float(ecliptic_lat_deg), float(solar_lon_deg))

    return ep.PlotResult(ax=ax, artists=panel.artists(), update=update)

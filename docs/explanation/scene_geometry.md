---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Scene geometry and views

A scene is three-dimensional, and every product made from it is a
projection: a detector image is the sky plane, a phase curve is a cut
through the star-planet-observer triangle, and a disk image is a
line-of-sight integral through an inclined dust layer. This page fixes the
frames skyscapes uses for that geometry and draws them with
{mod}`skyscapes.viz`, which needs the `viz` extra:

```bash
pip install 'skyscapes[viz]'
```

Every figure below is drawn from the same synthetic system, built on the
page, so each view can be checked against the others.

```{code-cell} ipython3
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hwoutils.constants import Mearth2kg, Msun2kg
from hwoutils.grids import PixelGrid, SkyOrientation
from zodi import SurfaceBrightnessDust
import equinox as eqx

from skyscapes import System, viz
from skyscapes.disk import GraterDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")

star = FlatStar(flux_phot_per_nm_m2=1e9)
planets = Planet(
    Rp_Rearth=jnp.array([1.0, 3.0]),
    physical_model=LambertianPhysicalModel(Ag=jnp.array([0.3, 0.3])),
)
disk = GraterDisk(
    sma_AU=jnp.array(3.0),
    alpha_in=jnp.array(5.0),
    alpha_out=jnp.array(-3.0),
    ksi0_AU=jnp.array(0.1),
    gamma=jnp.array(2.0),
    beta=jnp.array(1.0),
    rmin_AU=jnp.array(1.5),
    rmax_AU=jnp.array(5.0),
    wavelengths_nm=jnp.array([400.0, 1000.0]),
    g_HG_grid=jnp.array([0.4, 0.4]),
    Ag_grid=jnp.array([0.3, 0.3]),
    amplitude=SurfaceBrightnessDust(level=3, reference_radius_au=3.0),
    n_slices_los=41,
)
grid = PixelGrid.regular((121, 121), 0.0085, "arcsec", orientation=SkyOrientation(0.0))
system = System.from_classical(
    star=star,
    planets=(planets,),
    star_mass_Msun=1.0,
    planet_mass_Msun=jnp.array([1.0, 10.0]) * Mearth2kg / Msun2kg,
    dist_pc=10.0,
    a_AU=jnp.array([1.0, 2.5]),
    e=jnp.array([0.05, 0.2]),
    W_rad=jnp.array([0.4, 0.6]),
    i_rad=jnp.deg2rad(jnp.array([55.0, 62.0])),
    w_rad=jnp.array([0.3, 1.2]),
    M0_rad=jnp.array([0.0, 2.0]),
    disk=disk,
    midplane_inc_deg=60.0,
    midplane_position_angle_deg=60.0,
)
```

(scene-frames)=
## The frames

| Quantity | Convention |
|---|---|
| Sky frame | `(x, y, z)`, star-centered. `x` is the east offset (Delta alpha* = Delta RA cos Dec), `y` the north (Dec) offset, and `+z` points away from the observer. This (east, north, away) frame is right-handed, with dz/dt positive receding. `System.ephemeris` holds barycentric states; `System.positions` returns star-relative `(x, y)` in arcsec. The phase angle is the standard star-planet-observer angle, 0 at full phase (planet behind the star) and pi at new phase, as orbix returns it, so `cos(beta) = z / r` in this frame. |
| Sky images | Pixel centers and orientation come from the caller's `PixelGrid`. The standard north-up grid has `SkyOrientation(0, parity=-1)`, east to the left. Other rotations, parity and pixel bases are explicit. |
| Disk orientation | `System.midplane_inc_deg` and `System.midplane_position_angle_deg` override parametric disk orientations. Loaded cubes retain their native geometry. |
| Inclination | Zero face-on; the projected axis ratio is `abs(cos(i))`. The representations `(i, PA)` and `(180-i, PA+180)` describe the same plane and near half. |
| Position angle | Astronomical, north through east: PA=0 places the major axis north-south, PA=90 east-west. |
| Near side | For inclination below 90, the near minor direction is `(-cos(PA), sin(PA))` in east/north; it reverses above 90. Grains toward the observer scatter forward. |
| Scattering angle | Zero forward scattering; the planetary illumination phase is its supplement. |

`plot_disk_radiance` displays photon radiance in east/north coordinates,
resampling an oriented Image to a north-up display grid. Bare arrays are
already north-up radiance and need an explicit pixel scale.

The ExoVista loader converts header `(I, PA)` to astronomical
`(I, (90+PA) % 360)`. Negating the writer's west and toward axes gives
nodes `(-cos(PA), sin(PA), 0)` and near vector
`(cos(I) sin(PA), cos(I) cos(PA), -sin(I))`. The converted angle preserves
that node line and the near vector. Header `(60,30)` therefore becomes
`(60,120)`, with the near half at astronomical PA 30. Stored images and
already on-sky orbital states are not rotated again.

## The system on the sky and side on

{func}`~skyscapes.viz.plot_system` draws a system in one physical frame.
The sky view is what a detector sees, in arcsec. The side view turns the
same scene so the line of sight runs left to right, with the observer to
the left (negative z); the vertical axis is the sky offset along the
disk's projected minor axis. The disk appears side on as its midplane, tilted from the sky
plane by the inclination, and each planet sits at its true distance
toward or away from the observer.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
track = jnp.linspace(0.0, 1500.0, 300)
viz.plot_system(system, 0.0, ax=axes[0], track_time_d=track)
viz.plot_system(system, 0.0, view="side", ax=axes[1], track_time_d=track);
```

The dashed line on the sky view is the line of nodes, at 30 degrees from
`+x` toward `+y`, which is an astronomical position angle of 60 degrees
east of north. The label marks the near side, which the side view puts on
the observer's side of the sky plane.

## One grain, two angles

{func}`~skyscapes.viz.plot_scattering_angle` draws the definition in the
table above at full size: one grain in the plane its incident and
scattered directions span, the star behind the incident ray, the observer
ahead of the scattered one, and the scattering angle measured from the
forward continuation of the incident ray (dashed). It is the drawing the
geometry views magnify in their insets, so a figure can introduce the two
angles on their own before placing the grain in a disk.

```{code-cell} ipython3
incl = np.radians(60.0)
fig, ax = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
viz.plot_scattering_angle([-np.sin(incl), np.cos(incl)], [-1.0, 0.0], ax=ax);
```

Here the incident ray reaches a near-side grain on the minor axis of a
disk inclined by 60 degrees, and the scattered ray leaves along the line
of sight, so the grain scatters at 30 degrees. Three-element directions
are also accepted and are laid into the plane they span.

`values="arcs"` prints the two values on the arc labels instead of in the
corners (`values=None` prints neither), and `theta_radius` and
`alpha_radius` set the arc radii in units of the ray length; a narrow
illumination wedge reads better on a larger arc. `gid_prefix="inset"`
gives the drawing the gids the geometry views' insets carry, so a figure
that introduces the grain on its own can restyle it with the same code
as the inset it later becomes.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
viz.plot_scattering_angle(
    [-np.sin(incl), np.cos(incl)],
    [-1.0, 0.0],
    values="arcs",
    alpha_radius=0.55,
    ax=ax,
);
```

## From the midplane to the projected image

{func}`~skyscapes.viz.plot_disk_geometry` draws the disk side on with one
sightline through one grain, and {func}`~skyscapes.viz.plot_disk_radiance`
renders the disk's `photon_radiance` on the sky. Passing both the same
`grain_radius_AU` marks the sightline's position on the image.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
viz.plot_disk_geometry(system, grain_radius_AU=3.0, thickness_AU=0.6, ax=axes[0])
viz.plot_disk_radiance(system, grid=grid, wavelength_nm=550.0, grain_radius_AU=3.0, ax=axes[1]);
```

A grain on the minor axis of the near side scatters at `90 - i` degrees,
here 30 degrees, and its far-side mirror at `90 + i`. With the
Henyey-Greenstein asymmetry of this disk, `g = 0.4`, the forward-scattered
near side is the brighter half of the image. The inset magnifies the grain
and sets the scattering angle beside the illumination angle, the
quantity a planetary phase function takes; passing one where the other is
expected exchanges forward and back scattering.

## Drawing part of a view

`plot_disk_geometry` and `plot_local_zodi_geometry` take `show=`, the
parts to draw, so a figure can build a view up one idea at a time with a
single constructor. The parts left out are created hidden and keep their
gids, so a later figure that shows them draws them from the same numbers.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
viz.plot_disk_geometry(
    system,
    grain_radius_AU=3.0,
    thickness_AU=0.6,
    show=("star", "sky_plane", "disk", "layer", "inclination", "observer"),
    ax=axes[0],
)
viz.plot_disk_geometry(system, grain_radius_AU=3.0, thickness_AU=0.6, ax=axes[1]);
```

The left panel is the tilted disk alone; the right adds the sightline, the
grain, its rays and the inset. Each view's docstring lists its part names.
The observer's mark and its label are separate parts, `"observer_mark"`
and `"observer_label"` (`"observer"` names both), so a figure that draws
its own observer can keep the label, renamed with `observer_text`. The
image view takes `show=` too, for its overlays: `"star"`, `"outline"` and
`"sightline"`. A small inset has room for its angle values on the arcs
(`inset_values="arcs"`) rather than in its corners.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
viz.plot_disk_geometry(
    system,
    grain_radius_AU=3.0,
    inset_values="arcs",
    observer_text="to the telescope",
    ax=axes[0],
)
viz.plot_disk_radiance(
    system,
    grid=grid,
    wavelength_nm=550.0,
    grain_radius_AU=3.0,
    show=("star", "outline"),
    ax=axes[1],
);
```

## Sweeping the inclination

Each view returns an `update`. The geometry panel's `update(incl_deg)`
moves the midplane, the grain and its rays; the image panel's
`update(image, incl_deg=...)` swaps in a new rendering and moves the
outline. The science runs once, before the animation: the frames are
rendered first and the updates only move artists.

```{code-cell} ipython3
import eyepiece as ep
from IPython.display import HTML

inclinations = jnp.linspace(10.0, 80.0, 12)
frames = [
    eqx.tree_at(lambda d: (d.inclination_deg, d.position_angle_deg), disk,
                (incl, jnp.array(60.0))).photon_radiance(
        jnp.array(550.0), jnp.array(0.0), star=star,
        dist_pc=system.dist_pc, grid=grid,
    ).data
    for incl in inclinations
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
geometry = viz.plot_disk_geometry(
    system, incl_deg=float(inclinations[0]), grain_radius_AU=3.0, ax=axes[0]
)
image = viz.plot_disk_radiance(
    frames[0],
    vmax=max(float(frame.max()) for frame in frames),
    pixel_scale_arcsec=grid.step,
    incl_deg=float(inclinations[0]),
    position_angle_deg=60.0,
    dist_pc=10.0,
    radii_AU=(1.5, 5.0),
    grain_radius_AU=3.0,
    colorbar="figure",
    ax=axes[1],
)


def draw(fig, k):
    geometry.update(float(inclinations[k]))
    image.update(frames[k], incl_deg=float(inclinations[k]))


animation = ep.animate(fig, draw, len(frames), fps=4)
HTML(animation.jshtml(dpi=70))
```

The color scale is pinned to the brightest frame and kept for every
frame, so the image visibly brightens as the disk tilts: the same dust is
packed into fewer pixels, and the near side swings toward forward
scattering.

## Local zodiacal light is a different geometry

The same scattering physics makes the local zodiacal light, but the
observer is inside that cloud rather than far outside it. The
{doc}`local zodi geometry <local_zodi_geometry>` page draws that case with
{func}`~skyscapes.viz.plot_local_zodi_geometry`.

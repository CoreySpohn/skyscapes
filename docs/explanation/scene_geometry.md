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
from hwoutils.constants import Msun2kg
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit

from skyscapes import System, viz
from skyscapes.disk import GraterDisk
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import FlatStar, Planet

jax.config.update("jax_enable_x64", True)
hwostyle.use("light")

star = FlatStar(Ms_kg=Msun2kg, dist_pc=10.0, flux_phot_per_nm_m2=1e9)
planets = Planet(
    Rp_Rearth=jnp.array([1.0, 3.0]),
    Mp_Mearth=jnp.array([1.0, 10.0]),
    orbit=KeplerianOrbit(
        a_AU=jnp.array([1.0, 2.5]),
        e=jnp.array([0.05, 0.2]),
        W_rad=jnp.array([0.4, 0.6]),
        i_rad=jnp.deg2rad(jnp.array([55.0, 62.0])),
        w_rad=jnp.array([0.3, 1.2]),
        M0_rad=jnp.array([0.0, 2.0]),
        t0_d=jnp.array([0.0, 0.0]),
    ),
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
    nx=121,
    ny=121,
    pixel_scale_arcsec=0.0085,
    dist_pc=10.0,
    n_slices_los=41,
)
system = System(
    star=star,
    planets=(planets,),
    trig_solver=get_grid_solver(level="scalar", E=False, trig=True, jit=True),
    disk=disk,
    midplane_inc_deg=60.0,
    midplane_pa_deg=30.0,
)
```

## The frames

| Quantity | Convention |
|---|---|
| Sky frame | `(x, y, z)`, star-centered. `x` is the RA offset, `y` the Dec offset, and `+z` points from the star toward the observer. `Planet.position_arcsec` returns `(x, y)` in arcsec, and orbix measures its phase angle from the `+z` axis. |
| Sky images | Pixel `(row, col)` lies at `(y, x)`; the RA offset is drawn increasing to the left. |
| Disk orientation | Set once, on the system, by `System.midplane_inc_deg` and `System.midplane_pa_deg`. Every disk component renders at that orientation; the planets' orbits carry their own elements and need not be coplanar with it. |
| Inclination | The angle between the midplane normal and the line of sight, in `[0, 180]`, 0 for a face-on disk. The projected axis ratio is `abs(cos(i))`, and `i` at `pa` is the same midplane as `180 - i` at `pa + 180`. The parametric disk kernels return a negated map above 90 degrees, so {func}`~skyscapes.viz.plot_disk_image` refuses such a map and names the equivalent orientation to render instead. |
| Position angle | The projected major axis (the line of nodes) lies at `pa` from `+x` toward `+y`, so `pa = 0` puts it along the RA axis. This is **not** the astronomical position angle: with `+x` east, the major axis lies at astronomical PA (north through east) `90 - pa`, modulo 180. |
| Near side | The half of the disk displaced toward `+z`. Its grains scatter forward, so a forward-scattering phase function brightens it. |
| Scattering angle | The angle at a grain between the incident propagation direction (star to grain) and the direction toward the observer; 0 is forward scattering. The illumination angle is its supplement. |

The views place the disk through the same rotation the ExoVista loader
applies and the line-of-sight kernels integrate in, so an outline drawn by
a view and a disk rendered by `surface_brightness` agree.

## The system on the sky and side on

{func}`~skyscapes.viz.plot_system` draws a system in one physical frame.
The sky view is what a detector sees, in arcsec. The side view turns the
same scene so the line of sight runs left to right, with the observer to
the right; the vertical axis is the sky offset along the disk's projected
minor axis. The disk appears side on as its midplane, tilted from the sky
plane by the inclination, and each planet sits at its true distance
toward or away from the observer.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
track = jnp.linspace(0.0, 1500.0, 300)
viz.plot_system(system, 0.0, ax=axes[0], track_t_jd=track)
viz.plot_system(system, 0.0, view="side", ax=axes[1], track_t_jd=track);
```

The dashed line on the sky view is the line of nodes, at 30 degrees from
`+x` toward `+y`, which is an astronomical position angle of 60 degrees
east of north. The label marks the near side, which the side view puts on
the observer's side of the sky plane.

## From the midplane to the projected image

{func}`~skyscapes.viz.plot_disk_geometry` draws the disk side on with one
sightline through one grain, and {func}`~skyscapes.viz.plot_disk_image`
renders the disk's `surface_brightness` on the sky. Passing both the same
`grain_radius_AU` marks the sightline's position on the image.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
viz.plot_disk_geometry(system, grain_radius_AU=3.0, thickness_AU=0.6, ax=axes[0])
viz.plot_disk_image(system, wavelength_nm=550.0, grain_radius_AU=3.0, ax=axes[1]);
```

A grain on the minor axis of the near side scatters at `90 - i` degrees,
here 30 degrees, and its far-side mirror at `90 + i`. With the
Henyey-Greenstein asymmetry of this disk, `g = 0.4`, the forward-scattered
near side is the brighter half of the image. The inset magnifies the grain
and sets the scattering angle beside the illumination angle, the
quantity a planetary phase function takes; passing one where the other is
expected exchanges forward and back scattering.

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
    disk.surface_brightness(jnp.array(550.0), jnp.array(0.0), incl, jnp.array(30.0))
    for incl in inclinations
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
geometry = viz.plot_disk_geometry(
    system, incl_deg=float(inclinations[0]), grain_radius_AU=3.0, ax=axes[0]
)
image = viz.plot_disk_image(
    frames[0],
    vmax=max(float(frame.max()) for frame in frames),
    pixel_scale_arcsec=disk.pixel_scale_arcsec,
    incl_deg=float(inclinations[0]),
    pa_deg=30.0,
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

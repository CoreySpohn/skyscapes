---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Local zodi + telescope geometry

The Leinert+1998 zodiacal-light surface brightness tables are indexed by
two angles, the ecliptic latitude of the line of sight and the
helio-ecliptic longitude difference between the line of sight and the
Sun. For a moving observer pointing at a fixed sky target, neither angle
is constant in any straightforward way, and the two are easy to confuse
because they coincide along the ecliptic plane but diverge elsewhere on
the sky. This page walks through the computation of those two angles
using `orbix.observatory.ecliptic_pointing`, the common ways the geometry
goes wrong, and the conventions `LeinertZodi`
expects.

## The two Leinert inputs

The position-dependent factor in Leinert+1998 Table 17 depends on the
ecliptic latitude of the look vector, written `β_⊙` in Leinert and
computed by `LeinertZodi.photon_radiance` from its sky frame, and on the helio-ecliptic
longitude difference between the look vector and the Sun, written
`Δλ_⊙` in Leinert and computed the same way. The ecliptic latitude
is essentially time-invariant for a fixed sky target because stellar
parallax is negligible at typical mission distances and obliquity drift
is sub-arcsecond per year. By contrast, the helio-ecliptic longitude
difference sweeps the full range from 0° (target behind the Sun) to
180° (target opposite the Sun) and back over the course of one year as
the Sun's apparent ecliptic longitude rotates through 360°.

Both angles come from one function,
{func}`orbix.observatory.ecliptic_pointing`, which takes the target's
right ascension and declination in degrees, the reference `Epoch` and a
time in days from it. A scene that carries a sky position supplies the
first three through `scene.sky_frame`, so a caller never computes the angles:
`LeinertZodi.photon_radiance(wavelength_nm, sky_frame, time_d)` takes the
frame and the time and derives both. A scene without a position (`ra_deg` is
`None`) raises a `ValueError` naming the missing position; `AYOZodi` and
`PrecomputedZodi` ignore the frame.

```python
import jax.numpy as jnp
from orbix.observatory import ecliptic_pointing
from orbix.system import SkyFrame

from skyscapes import LeinertZodi

zodi = LeinertZodi()
frame = SkyFrame(10.0, ra_deg=100.0, dec_deg=-20.0)
wavelength_nm = jnp.array([500.0, 600.0])

flux = zodi.photon_radiance(wavelength_nm, frame, 120.0)  # ph/s/m^2/nm/sr

# The angles the model derives, for a sightline plot or a keep-out check:
ecl_lat, helio_lon = ecliptic_pointing(frame.ra_deg, frame.dec_deg, frame.epoch, 120.0)
```

{func}`skyscapes.viz.plot_local_zodi_geometry` draws what the two inputs
mean. Seen from the ecliptic pole, the sightline's projection leaves the
observer at the helio-ecliptic longitude difference from the Sun
direction; seen in the sightline's own vertical plane, it rises at the
ecliptic latitude. The brightness integrates along the half-ray from the
observer, and the inset shows, for one grain on that ray, the scattering
angle between the sunlight it receives and the light it sends back. The
readout under each panel gives the 3D solar elongation too, which is 128
degrees here against a 135 degree longitude difference, the distinction
the next section is about.

```{code-cell} ipython3
import hwostyle
import matplotlib.pyplot as plt

from skyscapes import viz

hwostyle.use("light")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
viz.plot_local_zodi_geometry(30.0, 135.0, ax=axes[0])
viz.plot_local_zodi_geometry(30.0, 135.0, view="side", ax=axes[1]);
```

Every grain on the sightline is lit by the Sun at its own distance, not
from the observer. Passing several distances marks several grains, each
with its own sunlight ray, and `show=` keeps the view to the parts a
figure needs; here the longitude arc, the readout and the scattered rays
are left out.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
viz.plot_local_zodi_geometry(
    30.0,
    135.0,
    grain_distance_AU=[0.4, 1.0, 1.4],
    show=("orbit", "sun", "observer", "sightline", "grain", "incident", "inset"),
    ax=ax,
);
```

The inset draws the first grain.

The longitude arc takes its label text as a template given the angle in
degrees (`look_angle_label`), its radius (`look_angle_radius_AU`) and, when
the automatic placement does not suit a figure, a fixed label position
about the observer (`look_angle_label_at`, an angle and a distance). The
observer's label can be kept while its mark is left out
(`"observer_label"` without `"observer_mark"`) and renamed
(`observer_text`).

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
viz.plot_local_zodi_geometry(
    30.0,
    135.0,
    look_angle_label="longitude difference\n= {value:.0f}$\\degree$",
    look_angle_radius_AU=0.45,
    look_angle_label_at=(100.0, 0.6),
    observer_text="telescope",
    show=(
        "orbit",
        "sun",
        "observer_label",
        "sightline",
        "grain",
        "incident",
        "look_angle",
        "inset",
    ),
    ax=ax,
);
```

The EXOSIMS documentation gives an equivalent walk-through of the
Leinert convention and the resulting brightness tables, and that
reference is helpful when cross-checking values against the original
Leinert paper.

## Why the 3D solar elongation is not the Leinert input

The observatory also exposes `ObservatoryL2Halo.solar_elongation_deg`, which
returns the great-circle angular distance between the Sun and the target
as seen from the observer. That quantity is what telescope field-of-regard
keep-out logic actually wants, but it is not the Leinert table input.
For an ecliptic-plane target the great-circle distance and the
helio-ecliptic longitude difference are mathematically identical, so the
two quantities agree, and either name happens to give the right
brightness. For high-latitude targets the two diverge by tens of
degrees, and feeding `solar_elongation_deg` into the Leinert lookup
returns the wrong brightness.

| Target | ecl_lat | 3D solar elongation | helio-ecliptic longitude | match? |
|---|---|---|---|---|
| (RA=0°, Dec=0°)   | +0°  | 179° | 179° | yes |
| (RA=90°, Dec=+23.44°) | +0°  | 89°  | 89°  | yes |
| (RA=0°, Dec=+60°) | +53° | 119° | 145° | **no** |
| (RA=0°, Dec=+80°) | +65° | 99°  | 113° | **no** |

Always use the helio-ecliptic longitude from `ecliptic_pointing` for
Leinert lookups. The keyword `helioecliptic_longitude_deg` names that
quantity and no other, which keeps the Leinert lookup correct at every
call site.

## The obliquity gotcha

A natural way to validate an annual-zodi pipeline is to pick four
targets equally spaced in right ascension and confirm that their
brightness curves peak at sequentially-shifted dates 90 days apart. That
argument is wrong as stated, because the ecliptic plane is tilted by
23.44° relative to the celestial equator, and only some equatorial
coordinates land on the ecliptic plane. A target at `(RA=90°, Dec=0°)`
sits at ecliptic latitude −23.44° and never has the Sun pass behind it,
so its brightness curve never peaks the way the other three targets'
curves do.

| Equatorial (RA, Dec) | Ecliptic (lon, lat) | On the ecliptic plane? |
|---|---|---|
| (0°, 0°)           | (0°, 0°)        | yes |
| (90°, 0°)          | (90°, **−23.44°**) | no |
| (90°, **+23.44°**) | (90°, 0°)       | yes |
| (180°, 0°)         | (180°, 0°)      | yes |
| (270°, **−23.44°**)| (270°, 0°)      | yes |

To validate the four-phases-of-the-year geometry, place each target on
the ecliptic plane by setting the declination to ±23.44° as appropriate
for the longitude. With those choices the four conjunction dates land
roughly 92 days apart, and the four peak amplitudes agree to within the
sampling cadence.

## Unobservability and the Leinert table edge

Near-Sun exclusion cells and out-of-range pointings are clamped by the
`zodi` library to its filled table. Radiance is finite there; telescope
keepout geometry must decide whether the target can be observed.

## See also

The `coronagraphoto` doc *Simulating zodi with a telescope orbit* shows
how to thread these two angles through the per-frame detector
simulation. {func}`orbix.observatory.ecliptic_pointing` documents the sky geometry
used above, and {class}`orbix.observatory.ObservatoryL2Halo` the L2 halo
orbit.
[Leinert+1998 A&A] contains the original tabulation, and the EXOSIMS
documentation provides an equivalent description of the table indexing
convention.

[Leinert+1998 A&A]: https://ui.adsabs.harvard.edu/abs/1998A%26AS..127....1L/abstract

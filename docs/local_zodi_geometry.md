# Local zodi + telescope geometry

The Leinert+1998 zodiacal-light surface brightness tables are indexed by
two angles, the ecliptic latitude of the line of sight and the
helio-ecliptic longitude difference between the line of sight and the
Sun. For a moving observer pointing at a fixed sky target, neither angle
is constant in any straightforward way, and the two are easy to confuse
because they coincide along the ecliptic plane but diverge elsewhere on
the sky. This page walks through the computation of those two angles
using `orbix.observatory.ObservatoryL2Halo`, the common ways the geometry
goes wrong, and the conventions `skyscapes.background.ZodiSourceLeinert`
expects.

## The two Leinert inputs

The position-dependent factor in Leinert+1998 Table 17 depends on the
ecliptic latitude of the look vector, written `β_⊙` in Leinert and
exposed as the `ecliptic_lat_deg` argument to
`ZodiSourceLeinert.spec_flux_density`, and on the helio-ecliptic
longitude difference between the look vector and the Sun, written
`Δλ_⊙` in Leinert and exposed as `solar_lon_deg`. The ecliptic latitude
is essentially time-invariant for a fixed sky target because stellar
parallax is negligible at typical mission distances and obliquity drift
is sub-arcsecond per year. By contrast, the helio-ecliptic longitude
difference sweeps the full range from 0° (target behind the Sun) to
180° (target opposite the Sun) and back over the course of one year as
the Sun's apparent ecliptic longitude rotates through 360°.

Both angles are read off the L2-halo geometry by
{class}`orbix.observatory.ObservatoryL2Halo`, which exposes them as
methods with consistent `(mjd, ra_rad, dec_rad)` argument order:

```python
from orbix.observatory import ObservatoryL2Halo
from skyscapes.background import ZodiSourceLeinert

obs = ObservatoryL2Halo.from_default()
zodi = ZodiSourceLeinert(reference_mag_arcsec2=22.0)

ecl_lat = obs.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)         # ~constant in time
helio_lon = obs.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)  # sweeps 0 -> 180 per year

flux = zodi.spec_flux_density(
    wavelength_nm,
    time_jd,
    ecliptic_lat_deg=ecl_lat,
    solar_lon_deg=helio_lon,
)
```

The EXOSIMS documentation gives an equivalent walk-through of the
Leinert convention and the resulting brightness tables, and that
reference is helpful when cross-checking values against the original
Leinert paper.

## Why the 3D solar elongation is not the Leinert input

The observatory module also exposes `solar_elongation_deg`, which
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

Always use `helio_ecliptic_longitude_deg` for Leinert lookups. The
explicit naming distinction is the reason the orbix API exposes both
methods rather than a single ambiguously-named one, and following that
convention at every call site keeps the Leinert lookup correct.

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

When the look vector points within a few degrees of the Sun, the
Leinert+1998 Table 17 lookup is out of range because the tabulation does
not extend to such tight elongations, and
`ZodiSourceLeinert.spec_flux_density` returns `NaN`. Downstream code
uses the `NaN` as a "target unobservable this epoch" gate, which is the
intended behaviour. Real telescope field-of-regard constraints typically
forbid pointing within 45° of the Sun, so the out-of-range region is
operationally never visited, and the `NaN` simply marks the geometric
impossibility of observing through the Sun.

## See also

The `coronagraphoto` doc *Simulating zodi with a telescope orbit* shows
how to thread these two angles through the per-frame detector
simulation. {class}`orbix.observatory.ObservatoryL2Halo` documents the
L2 halo orbit interpolator and the sky-geometry helpers used above.
[Leinert+1998 A&A] contains the original tabulation, and the EXOSIMS
documentation provides an equivalent description of the table indexing
convention.

[Leinert+1998 A&A]: https://ui.adsabs.harvard.edu/abs/1998A%26AS..127....1L/abstract

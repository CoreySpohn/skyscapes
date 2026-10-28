# Get the thermal-emission spectrum of a planet

{class}`~skyscapes.physical_model.ExoJaxEmissionModel` is the thermal-infrared
counterpart of {class}`~skyscapes.physical_model.ExoJaxPhysicalModel`. It
shares the species records, HITRAN opacity engines and mixing-ratio profiles
of the reflected-light model. Pure-absorption transfer is integrated locally
over an opaque, emitting ground using ExoJAX's pressure grid, opacity and
Gauss-Legendre quadrature over mu. The integration is two-stream-free and
includes no scattering; it does not call ExoJAX's transfer solver. It needs the
HITRAN line lists for each molecule in the ExoJAX cache (they are downloaded
on first use).

Ozone and sulfur dioxide use cross-section tables in the reflected-light
model, and those tables stop short of the infrared. The thermal builder always
uses HITRAN lines, so the 9.6 micron ozone band is present.

Atmospheric layer boundaries span `pressure_top_bar` to exactly
`surface_pressure_bar`. Opacity and atmospheric temperatures use the geometric
layer centers; the default ground temperature uses the profile at the exact
surface boundary. The hydrostatic column mass is the pressure difference
divided by gravity (Pierrehumbert 2010, *Principles of Planetary Climate*, Ch. 4).
The formal transfer solution and angular flux integral follow Rybicki and
Lightman (1979), *Radiative Processes in Astrophysics*, Sec. 1.4.

`flux_density_jy` uses linear interpolation in wavenumber to preserve
nonnegative flux. Requested wavelengths outside the engine's spectral grid
return NaN, including under JIT.

## An Earth twin at 10 pc

The temperature-pressure profile is the polynomial
$T(P) = \sum_i a_i (\log_{10} P[\mathrm{bar}])^i$ of
{class}`~skyscapes.physical_model.exojax.PolynomialTPProfile`; the
coefficients below describe a cloud-free Earth twin (Alei et al. 2024).

```python
import jax
import jax.numpy as jnp

from skyscapes.physical_model import ExoJaxEmissionModel

jax.config.update("jax_enable_x64", True)  # ExoJAX requires 64-bit floats

model = ExoJaxEmissionModel.from_default_setup(
    log_mmrs={
        "H2O": jnp.array([-3.0]),
        "CO2": jnp.array([-3.387]),
        "O3": jnp.array([-6.523]),
        "CH4": jnp.array([-5.770]),
        "N2O": jnp.array([-6.495]),
    },
    temperature_coefficients=jnp.array([[285.218, 146.626, 99.703, 23.120, 1.674]]),
    log_gravity_cgs=jnp.array([2.9917]),
    surface_pressure_bar=10**0.006,
)

wavelength_nm = jnp.linspace(4000.0, 18500.0, 1000)
flux_jy = model.flux_density_jy(
    wavelength_nm, Rp_Rearth=jnp.array([1.0]), distance_pc=10.0
)  # shape (W, K)
tb = model.brightness_temperature()  # shape (K, n_nu), on model.nu_grid
```

The 11 micron window sees the 286 K ground, the 15 micron CO2 band sees the
cold tropopause near 216 K, and the 9.6 micron ozone band sits in between.

## Collision-induced absorption

CIA is optional: `cia_pairs=()` leaves the line-only spectrum unchanged.
To include the O2-O2 fundamental near 6.4 microns and N2-N2 absorption, add
these arguments to `from_default_setup` above:

```python
cia_pairs=("O2-O2", "N2-N2"),
cia_vmr={"N2": 0.78, "O2": 0.21},
mean_molecular_weight=28.97,
```

`cia_vmr` specifies constant **volume** mixing ratios for partners absent
from `log_mmrs`. A modeled partner always uses its existing layer-dependent
mass-mixing-ratio profile, converted as `VMR = MMR * mean_molecular_weight /
molmass`; a `cia_vmr` entry cannot override that profile. Missing partners
raise an error. The mean molecular weight [u] is constant across layers and
planets and defaults to dry air; set it for a different bulk composition.
CIA uses `k_CIA * n_a * n_b * dz`, with coefficients in cm^5 molecule^-2,
number densities in cm^-3, and hydrostatic layer thickness in cm. There is
no extra symmetry factor for identical partners.

Place the desired [HITRAN CIA](https://hitran.org/cia/) files in
`~/.cache/skyscapes/cia/`, or supply `cia_dir`:

| Pair | Cached filename |
| --- | --- |
| N2-N2 | `N2-N2_2021.cia` |
| O2-O2 | `O2-O2_2024.cia` |
| O2-N2 | `O2-N2_2024.cia` |
| CO2-CO2 | `CO2-CO2_2024.cia` |
| N2-H2O | `N2-H2O_2018.cia` |

CIA files are never downloaded automatically. A missing file raises
`FileNotFoundError` with its expected path and the HITRAN link. For reusable
engines, pass `cia_pairs` and `cia_dir` to `build_exojax_emission_engines`,
then `cia_vmr` and `mean_molecular_weight` to `from_engines`.

The HITRAN CIA set is described by Karman et al. (2019), *Icarus* **328**,
160-175. ExoJAX 2.5's native reader handles N2-H2O; a local block reader
handles the other files, whose grids vary between bands and temperatures.
Coefficients are interpolated linearly in wavenumber and temperature,
with zero absorption outside tabulated spectral blocks. Negative experimental
baseline samples are floored at zero before interpolation. Temperatures outside
a band's tabulated range use its nearest endpoint. This avoids ExoJAX
`OpaCIA`'s next-bin spectral lookup and preserves positive tabulated values.
Blocks overlapping by at least half the narrower spectral interval are treated
as measurements of the same band. At repeated temperatures, later measurements
replace earlier ones only within their own coverage; adjacent bands remain
separate and add where their edges overlap.

## Reflected light on the same axis

`ExoJaxPhysicalModel.contrast_cube` returns planet-to-star flux ratios, with
wavelength first. Multiplying by the stellar flux density in Jy puts reflected
and thermal light on one axis. Both models accept wavelengths from the
ultraviolet to the infrared when built over that range
(`wavelength_min_nm` and `wavelength_max_nm`); a single reflected model
spans 200 to 2000 nm.

## What is left out

- There is no scattering (clouds and Rayleigh) in the thermal solver.
- The planet is a uniformly emitting disk: no day-night contrast and no phase
  dependence.
- A ground with emissivity below one (`surface_emissivity`) does not reflect
  the downwelling atmospheric radiation.

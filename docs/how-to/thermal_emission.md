# Get the thermal-emission spectrum of a planet

{class}`~skyscapes.physical_model.ExoJaxEmissionModel` is the thermal-infrared
counterpart of {class}`~skyscapes.physical_model.ExoJaxPhysicalModel`. It
shares the species records, HITRAN opacity engines and mixing-ratio profiles
of the reflected-light model, and replaces the 2-stream solver with ExoJAX's
pure-absorption emission solver over an opaque, emitting ground. It needs the
HITRAN line lists for each molecule in the ExoJAX cache (they are downloaded
on first use).

Ozone and sulfur dioxide use cross-section tables in the reflected-light
model, and those tables stop short of the infrared. The thermal builder always
uses HITRAN lines, so the 9.6 micron ozone band is present.

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

## Reflected light on the same axis

`ExoJaxPhysicalModel.contrast_cube` returns planet-to-star flux ratios, with
wavelength first. Multiplying by the stellar flux density in Jy puts reflected
and thermal light on one axis. Both models accept wavelengths from the
ultraviolet to the infrared when built over that range
(`wavelength_min_nm` and `wavelength_max_nm`); a single reflected model
spans 200 to 2000 nm.

## What is left out

- Collision-induced absorption is not included. It is a small effect for an
  Earth twin and matters mainly for dense CO2 atmospheres.
- There is no scattering (clouds and Rayleigh) in the thermal solver.
- The planet is a uniformly emitting disk: no day-night contrast and no phase
  dependence.
- A ground with emissivity below one (`surface_emissivity`) does not reflect
  the downwelling atmospheric radiation.
